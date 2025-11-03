from transformers import Owlv2ForObjectDetection
from torchcodec.decoders import VideoDecoder


def main():
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-large-patch14-ensemble", device_map="auto"
    )


if __name__ == "__main__":
    main()
