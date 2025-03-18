"""
有一个问题啊，如果ASD识别到有人在说话了，但是ASR没有检测到结果怎么办
    - 这里就当这个时刻没人说话，ASD识别结果可能有误。如果这样有问题，后面再处理
      (这个可能跟ASR模型的效果关系很大)
"""

import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)

from types import SimpleNamespace

import numpy as np
import rospy
from streaming_sensevoice import StreamingSenseVoice

from active_speaker_detection.msg import ActiveSpeakerAudio
from streaming_speech_recognition.msg import ASRResult


class ASR:
    def __init__(self):
        # others
        self.alg_args = SimpleNamespace(
            chunk_size=10,
            language="zh",
            device="cuda:0",
            is_last=True,
        )
        self.text = ""
        # model init
        self.model = StreamingSenseVoice(
            chunk_size=self.alg_args.chunk_size,
            language=self.alg_args.language,
            device=self.alg_args.device,
        )
        # ROS init
        rospy.Subscriber(
            "/ASD/ASD_result", ActiveSpeakerAudio, self._do_ASR, queue_size=10
        )
        self.pub_ASR_result = rospy.Publisher("ASR_result", ASRResult, queue_size=10)

    def _do_ASR(self, active_speaker_audio):
        ASR_result = ASRResult()
        ASR_result.seq_id = active_speaker_audio.seq_id
        ASR_result.track_id = active_speaker_audio.track_id
        if active_speaker_audio.tracker_id == -1:
            self.pub_ASR_result.publish(ASR_result)
        else:
            audio = np.array(active_speaker_audio.audio, dtype=np.int16)
            text = ""
            ASR_text = ""
            for res in self.model.streaming_inference(
                audio, is_last=self.alg_args.is_last
            ):
                text = res["text"]
            if len(text) == self.text:
                ASR_text = ""
                ASR_result.track_id = (
                    -1
                )  # 如果识别不出结果，那就先当这个人没有说话，后面可以再优化，先试试理想情况下方法效果怎么样
            else:
                ASR_text = text[len(self.text) :]
            self.text = text
            ASR_result.ASR_result = ASR_text
            self.pub_ASR_result.publish(ASR_result)


def main():
    rospy.init_node("streaming_ASR")
    ASR_model = ASR()
    rospy.spin()


if __name__ == "__main__":
    main()
