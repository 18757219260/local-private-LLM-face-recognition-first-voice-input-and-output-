import asyncio
import os
import sys
import time
import threading
import pyaudio
import edge_tts
import subprocess
import logging
import re
import webrtcvad
import array
import math
from io import BytesIO
from aip import AipSpeech

# 百度API配置
APP_ID = '118613302'
API_KEY = '7hSl10mvmtaCndZoab0S3BXQ'
SECRET_KEY = 'Fv10TxiFLmWb4UTAdLeA2eaTIE56QtkW'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class ImprovedInterruptibleTTS:
    def __init__(self, voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%"):
        # TTS配置
        self.voice = voice
        self.rate = rate
        self.volume = volume
        
        # 中断控制
        self.is_speaking = False
        self.should_interrupt = False
        self.listen_thread = None
        self.playback_process = None
        
        # 音频输入配置
        self.CHUNK = 320  # 20ms at 16kHz
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # 初始采样率，可能在设置过程中调整
        
        # 尝试两种不同的VAD设置
        try:
            # 主VAD设置 - 中等灵敏度
            self.vad = webrtcvad.Vad(2)  
            # 备用VAD - 更高灵敏度，用于额外检测
            self.vad_sensitive = webrtcvad.Vad(3)
            logging.info("VAD已初始化，使用双重检测模式")
        except Exception as e:
            logging.error(f"初始化VAD失败: {e}")
            self.vad = None
            self.vad_sensitive = None
        
        # 初始化PyAudio
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        
        # 添加麦克风设置
        self.setup_best_microphone()
    
    def update_vad_frame_size(self):
        """根据当前采样率更新VAD处理的帧大小
        
        WebRTC VAD需要特定长度的帧，具体取决于采样率:
        - 8000Hz: 80, 160, 240字节帧 (10, 20, 30毫秒)
        - 16000Hz: 160, 320, 480字节帧 (10, 20, 30毫秒)
        - 32000Hz: 320, 640, 960字节帧 (10, 20, 30毫秒)
        - 48000Hz: 480, 960, 1440字节帧 (10, 20, 30毫秒)
        """
        try:
            # 基于采样率选择合适的帧大小 (我们使用20ms帧)
            if self.RATE == 8000:
                self.CHUNK = 160
            elif self.RATE == 16000:
                self.CHUNK = 320
            elif self.RATE == 32000:
                self.CHUNK = 640
            elif self.RATE == 48000:
                self.CHUNK = 960
            else:
                # 对于其他采样率，使用最接近的标准值
                standard_rates = [8000, 16000, 32000, 48000]
                closest_rate = min(standard_rates, key=lambda x: abs(x - self.RATE))
                if closest_rate == 8000:
                    self.CHUNK = 160
                elif closest_rate == 16000:
                    self.CHUNK = 320
                elif closest_rate == 32000:
                    self.CHUNK = 640
                else:  # 48000Hz
                    self.CHUNK = 960
                    
            logging.info(f"已调整VAD帧大小: 采样率 {self.RATE}Hz, 帧大小 {self.CHUNK} 字节 (约20ms)")
            
            # 如果VAD实例存在，重新创建它们以确保适配新的采样率
            if hasattr(self, 'vad') and self.vad:
                aggressiveness = self.vad.get_mode()
                self.vad = webrtcvad.Vad(aggressiveness)
                logging.info(f"已重新初始化主VAD，灵敏度: {aggressiveness}")
                
            if hasattr(self, 'vad_sensitive') and self.vad_sensitive:
                self.vad_sensitive = webrtcvad.Vad(3)  # 敏感VAD总是使用最高级别
                logging.info("已重新初始化敏感VAD")
                
        except Exception as e:
            logging.error(f"更新VAD帧大小时出错: {e}")
            # 使用安全的默认值
            self.CHUNK = 320
        
    def setup_best_microphone(self):
        """选择最佳麦克风并配置输入设备，确保选择支持正确采样率的设备"""
        try:
            info = self.p.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            # 查找可用的输入设备
            input_devices = []
            default_input_device = None
            pulse_device = None  # 专门查找 pulse 设备，它通常支持更多采样率
            
            logging.info("开始查找音频输入设备...")
            
            for i in range(num_devices):
                device_info = self.p.get_device_info_by_host_api_device_index(0, i)
                
                if device_info.get('maxInputChannels') > 0:
                    device_name = device_info.get('name')
                    device_index = device_info.get('index')
                    
                    # 获取支持的采样率
                    try:
                        # 尝试确定设备支持的采样率
                        supported_rates = []
                        test_rates = [8000, 11025, 16000, 22050, 44100, 48000]
                        
                        for rate in test_rates:
                            try:
                                test_stream = self.p.open(
                                    format=pyaudio.paInt16,
                                    channels=1,
                                    rate=rate,
                                    input=True,
                                    input_device_index=device_index,
                                    frames_per_buffer=320,
                                    start=False  # 不实际启动流
                                )
                                test_stream.close()
                                supported_rates.append(rate)
                            except:
                                pass
                        
                        if supported_rates:
                            logging.info(f"设备 [{device_index}] {device_name} 支持的采样率: {supported_rates}")
                        else:
                            logging.warning(f"设备 [{device_index}] {device_name} 可能不支持任何标准采样率")
                        
                        # 仅添加支持我们需要采样率的设备
                        input_devices.append((device_index, device_name, supported_rates))
                        
                        # 记录特殊设备
                        if device_info.get('isDefaultInputDevice'):
                            default_input_device = (device_index, device_name, supported_rates)
                        if 'pulse' in device_name.lower():
                            pulse_device = (device_index, device_name, supported_rates)
                            
                    except Exception as e:
                        logging.warning(f"测试设备 [{device_index}] {device_name} 的采样率时出错: {e}")
                        # 如果无法测试采样率，仍然添加设备但标记为未知支持
                        input_devices.append((device_index, device_name, ['unknown']))
                        
                        # 记录特殊设备
                        if device_info.get('isDefaultInputDevice'):
                            default_input_device = (device_index, device_name, ['unknown'])
                        if 'pulse' in device_name.lower():
                            pulse_device = (device_index, device_name, ['unknown'])
            
            # 优先选择麦克风设备
            if input_devices:
                logging.info(f"找到 {len(input_devices)} 个音频输入设备:")
                for idx, name, rates in input_devices:
                    logging.info(f"  [{idx}] {name} - 支持的采样率: {rates}")
                
                # 选择设备的逻辑：优先选择 pulse 设备
                selected_device = None
                
                # 1. 首先尝试使用 pulse 设备，因为它几乎总是能工作
                if pulse_device:
                    selected_device = pulse_device
                    logging.info(f"选择 pulse 音频设备: [{selected_device[0]}] {selected_device[1]}")
                
                # 2. 如果没有 pulse 设备，尝试使用内置麦克风
                elif not selected_device:
                    for idx, name, rates in input_devices:
                        if ('内置' in name or 'internal' in name.lower() or 'mic' in name.lower()):
                            selected_device = (idx, name, rates)
                            logging.info(f"选择内置麦克风: [{selected_device[0]}] {selected_device[1]}")
                            break
                
                # 3. 如果上述都没有，使用默认设备
                if not selected_device and default_input_device:
                    selected_device = default_input_device
                    logging.info(f"选择默认音频设备: [{selected_device[0]}] {selected_device[1]}")
                
                # 4. 最后，使用任何可用设备
                if not selected_device and input_devices:
                    selected_device = input_devices[0]
                    logging.info(f"选择首个可用音频设备: [{selected_device[0]}] {selected_device[1]}")
                
                if selected_device:
                    self.input_device_index = selected_device[0]
                    # 保存设备支持的采样率，以便后续使用
                    if isinstance(selected_device[2], list) and len(selected_device[2]) > 0 and selected_device[2][0] != 'unknown':
                        if 16000 in selected_device[2]:
                            self.RATE = 16000  # 首选16kHz
                        else:
                            # 否则选择最接近16kHz的采样率
                            rates = sorted(selected_device[2])
                            closest_rate = min(rates, key=lambda x: abs(x-16000))
                            self.RATE = closest_rate
                            logging.info(f"设备不支持16kHz，使用最接近的采样率: {self.RATE}Hz")
                    
                    # 更新VAD帧大小
                    self.update_vad_frame_size()
                    logging.info(f"已选择输入设备: [{selected_device[0]}] {selected_device[1]} 采样率: {self.RATE}Hz")
                    return True
            
            logging.warning("未找到合适的输入设备，将尝试使用系统默认设备")
            self.input_device_index = None
            # 尝试使用较低的采样率
            self.RATE = 8000  # 降低采样率尝试兼容性
            self.update_vad_frame_size()
            logging.info(f"使用较低的采样率: {self.RATE}Hz 以提高兼容性")
            return False
        
        except Exception as e:
            logging.error(f"设置麦克风时出错: {e}")
            self.input_device_index = None
            # 尝试使用较低的采样率
            self.RATE = 8000
            self.update_vad_frame_size()
            logging.info(f"出错后使用较低的采样率: {self.RATE}Hz")
            return False
            
    def preprocess_text(self, text):
        """预处理文本，替换标点符号"""
        if not text:
            return ""
            
        text = text.replace("，", ",")
        text = text.replace("。", ",")
        text = text.replace("、", ",")
        text = text.replace("；", ",")
        text = text.replace("：", ",")
        text = text.replace("*", ',')
        text = text.replace(".", ',')
        text = text.replace("#", ',')
        text = text.replace("？", ',')
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        return text
        
    def setup_input_stream(self):
        """设置音频输入流，具有更强的错误处理和降级策略"""
        if self.input_stream is None:
            # 尝试多种采样率，从高到低
            fallback_rates = [16000, 44100, 48000, 8000, 11025, 22050]
            original_rate = self.RATE  # 保存原始采样率
            
            # 如果当前采样率不在候选列表中，添加它
            if self.RATE not in fallback_rates:
                fallback_rates.insert(0, self.RATE)
            
            # 记录当前尝试的采样率索引
            rate_index = 0
            max_attempts = len(fallback_rates)
            success = False
            
            while rate_index < max_attempts and not success:
                try:
                    self.RATE = fallback_rates[rate_index]
                    logging.info(f"尝试以采样率 {self.RATE}Hz 打开输入流...")
                    
                    # 使用选定的输入设备
                    kwargs = {
                        "format": self.FORMAT,
                        "channels": self.CHANNELS,
                        "rate": self.RATE,
                        "input": True,
                        "frames_per_buffer": self.CHUNK
                    }
                    
                    # 如果有指定输入设备，则使用它
                    if hasattr(self, 'input_device_index') and self.input_device_index is not None:
                        kwargs["input_device_index"] = self.input_device_index
                    
                    # 尝试打开输入流
                    self.input_stream = self.p.open(**kwargs)
                    
                    # 检验流是否真的打开
                    if self.input_stream.is_active():
                        logging.info(f"成功打开输入流，使用采样率: {self.RATE}Hz")
                        success = True
                        
                        # 读取一些数据以确保流能正常工作
                        test_data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                        logging.info(f"成功读取测试音频数据，长度: {len(test_data)} 字节")
                        
                        # 调整VAD模型帧大小以适应采样率
                        self.update_vad_frame_size()
                        
                        # 测试麦克风音量
                        self.test_microphone_levels()
                        
                        return True
                    else:
                        logging.warning(f"输入流创建但未激活，尝试下一个采样率")
                        self.input_stream.close()
                        self.input_stream = None
                        
                except Exception as e:
                    logging.error(f"使用采样率 {self.RATE}Hz 设置输入流出错: {e}")
                    if self.input_stream:
                        try:
                            self.input_stream.close()
                        except:
                            pass
                        self.input_stream = None
                    
                # 尝试下一个采样率
                rate_index += 1
            
            # 如果所有尝试都失败，回退到原始采样率
            if not success:
                self.RATE = original_rate
                logging.error(f"无法找到工作的采样率，回退到: {self.RATE}Hz")
                
                # 尝试使用默认设备（最后的努力）
                try:
                    logging.info("尝试使用系统默认输入设备...")
                    self.input_stream = self.p.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK
                    )
                    logging.info("成功使用系统默认设备打开输入流")
                    self.update_vad_frame_size()
                    return True
                except Exception as e:
                    logging.error(f"使用系统默认设备也失败: {e}")
                    return False
        
        return True  # 如果输入流已经存在
        
    def test_microphone_levels(self):
        """测试麦克风音量级别"""
        if not self.input_stream:
            return
        
        try:
            logging.info("测试麦克风音量级别...")
            frames = []
            for _ in range(10):  # 收集约0.2秒的音频
                data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            # 计算音量
            audio_data = b''.join(frames)
            # 将字节转换为short整数
            shorts = array.array('h', audio_data)
            
            # 计算RMS值
            sum_squares = sum(s*s for s in shorts)
            rms = math.sqrt(sum_squares / len(shorts) if shorts else 1)
            
            # 计算分贝值（相对于最大值）
            max_possible = 32768.0  # 16位音频的最大值
            db = 20 * math.log10(rms / max_possible) if rms > 0 else -100
            
            logging.info(f"麦克风音量: {rms:.2f} RMS ({db:.2f} dB)")
            
            # 根据音量给出建议
            if db < -50:
                logging.warning("麦克风音量非常低，这可能导致语音检测问题。请检查麦克风设置或靠近麦克风说话。")
            elif db < -35:
                logging.info("麦克风音量较低，语音检测可能不够灵敏。")
            elif db > -15:
                logging.info("麦克风音量较高，可能会导致过度触发。")
            else:
                logging.info("麦克风音量正常。")
                
        except Exception as e:
            logging.error(f"测试麦克风级别时出错: {e}")
            
        # 创建专用的测试方法来检查语音检测
        self.test_speech_detection()
        
    def test_speech_detection(self):
        """测试语音检测功能"""
        if not self.input_stream or not self.vad:
            return
            
        try:
            logging.info("测试VAD语音检测功能...")
            # 连续监听几帧，看看是否能检测到语音
            speech_frames = 0
            total_frames = 30  # 监听约0.6秒
            
            for _ in range(total_frames):
                data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                # 使用两个不同灵敏度的VAD测试
                is_speech_normal = self.vad.is_speech(data, self.RATE)
                is_speech_sensitive = self.vad_sensitive.is_speech(data, self.RATE) if self.vad_sensitive else False
                
                if is_speech_normal:
                    speech_frames += 1
                    
            speech_ratio = speech_frames / total_frames
            logging.info(f"语音检测测试: 检测到 {speech_frames}/{total_frames} 帧包含语音 ({speech_ratio:.2%})")
            
            # 提供诊断信息
            if speech_ratio > 0.5:
                logging.warning("检测到大量背景噪音，这可能导致误触发。考虑使用降噪麦克风或在安静环境中使用。")
            elif speech_ratio > 0.1:
                logging.info("检测到少量背景噪音，但VAD应该能够正常工作。")
            else:
                logging.info("环境安静，建议进行语音测试以确保VAD正确工作。")
                
        except Exception as e:
            logging.error(f"测试语音检测时出错: {e}")
    
    def close_input_stream(self):
        """关闭音频输入流"""
        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None
                logging.info("输入流已关闭")
            except Exception as e:
                logging.error(f"关闭输入流出错: {e}")
    
    def listen_for_interruption(self):
        """监听用户语音，检测是否需要中断"""
        if not self.setup_input_stream():
            logging.error("无法设置输入流进行中断监听")
            return
            
        consecutive_speech_frames = 0
        required_speech_frames = 3  # 降低触发中断所需的连续语音帧数
        silence_frames = 0
        speech_ratio = 0.0
        frame_history = []
        history_size = 20  # 保存最近20帧用于计算语音比例
        
        logging.info("开始监听中断...")
        
        while self.is_speaking:
            try:
                data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                is_speech = self.vad.is_speech(data, self.RATE)
                
                # 保存帧历史用于计算语音比例
                frame_history.append(1 if is_speech else 0)
                if len(frame_history) > history_size:
                    frame_history.pop(0)
                
                # 计算最近帧中的语音比例
                speech_ratio = sum(frame_history) / len(frame_history)
                
                if is_speech:
                    consecutive_speech_frames += 1
                    silence_frames = 0
                    
                    # 使用双重条件触发中断:
                    # 1. 连续语音帧达到阈值 或
                    # 2. 语音比例超过40%
                    if consecutive_speech_frames >= required_speech_frames or speech_ratio > 0.4:
                        logging.info(f"检测到用户语音，准备中断... (连续帧数: {consecutive_speech_frames}, 语音比例: {speech_ratio:.2f})")
                        self.should_interrupt = True
                        # 立即终止播放进程
                        self.stop_playback()
                        break
                else:
                    silence_frames += 1
                    if silence_frames > 2:  # 降低重置计数器的帧数
                        consecutive_speech_frames = max(0, consecutive_speech_frames - 1)  # 平滑降低而不是直接重置
                    
            except Exception as e:
                logging.error(f"监听中断时出错: {e}")
                break
                
        logging.info("停止监听中断")
    
    def stop_playback(self):
        """停止正在播放的音频"""
        if self.playback_process and self.playback_process.poll() is None:
            try:
                # 尝试终止播放进程
                self.playback_process.terminate()
                # 给进程一点时间终止
                time.sleep(0.2)
                # 如果还没终止，强制杀死
                if self.playback_process.poll() is None:
                    self.playback_process.kill()
                logging.info("音频播放已停止")
            except Exception as e:
                logging.error(f"停止音频播放出错: {e}")
    
    async def get_user_input(self):
        """获取用户输入（打断后）"""
        logging.info("请问您有什么需要？")
        
        # 确保使用正确的输入流
        if not self.setup_input_stream():
            logging.error("无法设置输入流获取用户输入")
            return None
        
        # 清空可能的音频缓冲
        try:
            time.sleep(0.5)
            while self.input_stream.get_read_available() > 0:
                self.input_stream.read(self.CHUNK, exception_on_overflow=False)
        except Exception as e:
            logging.error(f"清空音频缓冲出错: {e}")
        
        # 配置录音参数
        input_frames = []
        start_time = time.time()
        speech_started = False
        last_speech_time = time.time()
        silence_duration = 1.0
        max_record_seconds = 7.0
        
        logging.info("请说话...")
        
        while True:
            try:
                data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                is_speech = self.vad.is_speech(data, self.RATE)
                
                if is_speech:
                    if not speech_started:
                        speech_started = True
                        logging.info("检测到语音输入...")
                    last_speech_time = time.time()
                    input_frames.append(data)
                else:
                    if speech_started and (time.time() - last_speech_time) >= silence_duration:
                        logging.info("语音输入结束")
                        break
                        
                if (time.time() - start_time) >= max_record_seconds:
                    logging.info("达到最大录音时间")
                    break
                    
            except Exception as e:
                logging.error(f"录音出错: {e}")
                break
        
        if input_frames:
            try:
                audio_data = b"".join(input_frames)
                logging.info(f"上传 {len(audio_data)} 字节到百度API")
                result = client.asr(audio_data, 'pcm', self.RATE, {'dev_pid': 1537})
                
                if result['err_no'] == 0:
                    user_question = result['result'][0]
                    logging.info(f"用户说: {user_question}")
                    return user_question
                else:
                    logging.error(f"识别失败: {result['err_msg']}, 错误码: {result['err_no']}")
                    return None
            except Exception as e:
                logging.error(f"语音识别API调用出错: {e}")
                return None
        else:
            logging.info("没有录到语音")
            return None
    
    async def analyze_background_noise(self):
        """分析背景噪音并动态调整VAD设置"""
        if not self.setup_input_stream():
            return
            
        try:
            logging.info("分析背景噪音...")
            speech_frames = 0
            total_frames = 50  # 监听约1秒
            
            for _ in range(total_frames):
                data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                if self.vad_sensitive and self.vad_sensitive.is_speech(data, self.RATE):
                    speech_frames += 1
                    
            noise_ratio = speech_frames / total_frames
            
            # 根据背景噪音水平调整VAD灵敏度
            if noise_ratio > 0.4:
                # 噪音较大环境，降低灵敏度，避免误触发
                self.vad = webrtcvad.Vad(1)
                logging.info(f"环境噪音较大 ({noise_ratio:.2%})，已调整至低灵敏度")
            elif noise_ratio > 0.2:
                # 适中噪音环境，使用中等灵敏度
                self.vad = webrtcvad.Vad(2)
                logging.info(f"环境噪音适中 ({noise_ratio:.2%})，已调整至中等灵敏度")
            else:
                # 安静环境，提高灵敏度以便捕捉轻声说话
                self.vad = webrtcvad.Vad(3)
                logging.info(f"环境安静 ({noise_ratio:.2%})，已调整至高灵敏度")
                
        except Exception as e:
            logging.error(f"分析背景噪音出错: {e}")
        # 不关闭输入流，因为稍后会用于监听中断
    
    async def prepare_audio_file(self, text):
        """准备音频文件，返回文件路径"""
        try:
            # 创建edge-tts通信实例
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume
            )
            
            # 生成唯一的临时文件名
            temp_file = f"temp_audio_{int(time.time())}_{id(self)}.mp3"
            
            # 保存音频到文件
            await communicate.save(temp_file)
            logging.info(f"音频文件已保存: {temp_file}")
            
            return temp_file
        except asyncio.CancelledError:
            logging.warning("准备音频文件时被取消")
            return None
        except Exception as e:
            logging.error(f"准备音频文件出错: {e}")
            return None
    
    async def play_audio_with_interrupt(self, audio_file):
        """播放音频文件，同时监听中断"""
        if not os.path.exists(audio_file):
            logging.error(f"音频文件不存在: {audio_file}")
            return False, None
        
        try:
            # 设置状态
            self.is_speaking = True
            self.should_interrupt = False
            
            # 启动监听线程
            self.listen_thread = threading.Thread(target=self.listen_for_interruption)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            
            # 启动播放进程
            self.playback_process = subprocess.Popen(
                ["mpg123", audio_file], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            # 监听中断事件
            was_interrupted = False
            while self.playback_process.poll() is None:  # 当进程仍在运行
                if self.should_interrupt:
                    self.stop_playback()
                    was_interrupted = True
                    logging.info("语音输出已被用户中断")
                    break
                await asyncio.sleep(0.1)  # 短暂休眠，减少CPU占用
            
            # 等待线程完成
            if self.listen_thread and self.listen_thread.is_alive():
                self.is_speaking = False
                self.listen_thread.join(timeout=1.0)
            
            # 检查是否播放完成还是被中断
            if was_interrupted:
                logging.info("处理中断...")
                await asyncio.sleep(0.5)  # 确保音频已停止
                
                # 播放"怎么了?"
                await self.text_to_speech("怎么了?")
                
                # 获取新的用户输入
                user_input = await self.get_user_input()
                return True, user_input
            else:
                logging.info("音频播放完成")
                return False, None
                
        except Exception as e:
            logging.error(f"播放音频出错: {e}")
            return False, None
        finally:
            # 重置状态和清理
            self.is_speaking = False
            self.should_interrupt = False
            self.playback_process = None
            
            # 尝试删除临时文件
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    logging.info(f"临时音频文件已删除: {audio_file}")
            except Exception as e:
                logging.error(f"删除临时文件出错: {e}")
    
    async def speak_with_interrupt(self, text):
        """带有中断功能的语音输出主函数 - 改进版"""
        processed_text = self.preprocess_text(text)
        if not processed_text:
            logging.warning("文本为空，跳过语音播放")
            return False, None
            
        logging.info("-" * 50)
        logging.info(f"开始播放: {processed_text[:100]}...")
        logging.info("-" * 50)
        
        # 分析环境噪音水平，动态调整VAD灵敏度
        await self.analyze_background_noise()
            
        try:
            # 先准备音频文件
            audio_file = await self.prepare_audio_file(processed_text)
            if not audio_file:
                logging.error("无法准备音频文件")
                return False, None
                
            # 播放并监听中断
            return await self.play_audio_with_interrupt(audio_file)
                
        except asyncio.CancelledError:
            logging.warning("语音播放操作被取消")
            return False, None
        except Exception as e:
            logging.error(f"语音播放出错: {e}")
            return False, None
        finally:
            # 确保关闭输入流
            self.close_input_stream()
    
    async def text_to_speech(self, text):
        """基本的文本到语音转换（不带中断功能，用于短句）"""
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return
            
        try:
            # 创建edge-tts通信实例
            communicate = edge_tts.Communicate(
                text=processed_text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume
            )
            
            # 生成临时文件
            output_file = f"temp_simple_{int(time.time())}_{id(self)}.mp3"
            await communicate.save(output_file)
            
            # 播放
            subprocess.run(
                ["mpg123", output_file], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            # 删除临时文件
            if os.path.exists(output_file):
                os.remove(output_file)
                
        except Exception as e:
            logging.error(f"简单语音播放失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        self.close_input_stream()
        self.stop_playback()
        
        if self.p:
            self.p.terminate()
            logging.info("已清理所有音频资源")


# 简单演示代码
async def demo():
    tts = ImprovedInterruptibleTTS()
    
    try:
        print("\n" + "="*60)
        print("演示可中断的语音输出")
        print("在听到语音输出时，请尝试说话来中断它")
        print("="*60 + "\n")
        
        long_text = """甘薯又名地瓜、红薯、番薯等，是世界上重要的粮食作物之一。它富含淀粉、膳食纤维、胡萝卜素、维生素C和矿物质，营养价值非常高。甘薯品种繁多，根据肉色可分为白心、黄心、橙心、紫心等多种类型。甘薯适应性强，抗逆性好，可在多种气候条件下种植。它不仅是重要的粮食作物，还是优质的饲料作物和工业原料。在许多国家，甘薯被广泛用于食品加工行业，制作淀粉、酒精、糖浆等产品。近年来，随着人们对健康食品的追求，甘薯消费量逐年增加，其经济价值和社会价值不断提升。"""
        
        print("将开始播放语音，请准备随时打断...")
        await asyncio.sleep(2)  # 给用户准备的时间
        
        # 播放长文本，并等待可能的中断
        was_interrupted, user_question = await tts.speak_with_interrupt(long_text)
        
        if was_interrupted and user_question:
            print(f"\n用户提问: {user_question}")
            # 这里可以调用QA模型处理用户问题
            answer = f"关于{user_question}的回答是：这是一个示例回答，实际应用中这里会调用您的QA模型。"
            
            # 继续对话
            await tts.speak_with_interrupt(answer)
        else:
            print("\n语音播放完成，没有被中断")
        
        print("\n演示结束")
            
    except KeyboardInterrupt:
        print("\n程序被手动中断")
    except Exception as e:
        print(f"\n演示过程中出错: {e}")
    finally:
        tts.cleanup()


# 为了兼容原有代码，创建InterruptibleTTS类作为ImprovedInterruptibleTTS的别名
class InterruptibleTTS(ImprovedInterruptibleTTS):
    pass


if __name__ == "__main__":
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n程序被手动中断")
    except Exception as e:
        print(f"\n程序出错: {e}")