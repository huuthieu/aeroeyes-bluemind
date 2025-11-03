import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium", app_title="OWLv2")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    import torch
    from transformers import AutoProcessor, Owlv2ForObjectDetection


    @mo.cache
    def load_model(name_or_path: str):
        model = Owlv2ForObjectDetection.from_pretrained(
            name_or_path, device_map="auto", dtype=torch.float16
        )
        model = torch.compile(model)
        processor = AutoProcessor.from_pretrained(name_or_path, use_fast=True)
        return model, processor


    @torch.inference_mode()
    def forward(
        model: Owlv2ForObjectDetection,
        query_pixel_values: torch.Tensor,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool,
    ):
        query_feature_map = model.image_embedder(
            query_pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )[0]

        target_feature_map = model.image_embedder(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )[0]

        batch_size = query_feature_map.size(0)
        output_dim = query_feature_map.size(-1)

        query_feats = query_feature_map.reshape(batch_size, -1, output_dim)
        objectness_logits = model.objectness_predictor(query_feats)
        best_box_indices = torch.argmax(objectness_logits, dim=1).view(-1, 1, 1)

        query_class_embeds = model.class_predictor(query_feats)[1]
        hidden_size = query_class_embeds.size(-1)
        # B 1 hidden_size
        query_embeds = torch.gather(
            query_class_embeds,
            dim=1,
            index=best_box_indices.expand(-1, 1, hidden_size),
        )

        target_feats = target_feature_map.reshape(batch_size, -1, output_dim)
        pred_boxes = model.box_predictor(
            target_feats,
            target_feature_map,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        scores = torch.sigmoid(
            model.class_predictor(target_feats, query_embeds)[0]
        )
        return pred_boxes, scores
    return forward, load_model


@app.cell
def _(mo):
    from pathlib import Path

    _DATA_PATH = Path("./data")
    video_dropdown = mo.ui.dropdown(
        options := {
            str(path.relative_to(_DATA_PATH)): path
            for path in _DATA_PATH.rglob("*.mp4")
        },
        value=list(options.keys())[0],
        label="Select video",
        searchable=True,
    )

    model_dropdown = mo.ui.dropdown(
        options := [
            "google/owlv2-base-patch16-ensemble",
            "google/owlv2-large-patch14-ensemble",
        ],
        value=options[0],
        label="Pretrain",
    )
    interpolate_pos_encoding_check = mo.ui.checkbox(
        value=True, label="interpolate_pos_encoding"
    )
    score_threshold_slider = mo.ui.slider(
        0,
        1,
        value=0.85,
        step=0.01,
        debounce=True,
        include_input=True,
        label="Score threshold",
    )

    mo.vstack(
        [
            video_dropdown,
            model_dropdown,
            interpolate_pos_encoding_check,
            score_threshold_slider,
        ]
    )
    return (
        interpolate_pos_encoding_check,
        model_dropdown,
        score_threshold_slider,
        video_dropdown,
    )


@app.cell
def _(mo, video_dropdown):
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video_path := video_dropdown.value)
    frame_slider = mo.ui.slider(
        0, len(decoder), debounce=True, include_input=True, label="Frame"
    )
    frame_slider
    return decoder, frame_slider, video_path


@app.cell
def pre_processing(
    decoder,
    frame_slider,
    interpolate_pos_encoding_check,
    load_model,
    model_dropdown,
    video_path,
):
    from torchvision.io import decode_image
    from transformers import TensorType

    name_or_path = model_dropdown.value
    interpolate_pos_encoding = interpolate_pos_encoding_check.value
    model, processor = load_model(name_or_path)

    _ref_dir = video_path.parent / "object_images"
    ref_img_paths = [ref_path for ref_path in _ref_dir.glob("*.jpg")]

    _ref_img_paths = [ref_img_paths[1]]
    query_inputs = processor(
        query_images=[
            decode_image(path, apply_exif_orientation=True)
            for path in _ref_img_paths
        ],
        return_tensors=TensorType.PYTORCH,
        device=model.device,
    )

    target_frame = decoder[frame_slider.value]
    target_inputs = processor(
        images=target_frame,
        return_tensors=TensorType.PYTORCH,
        device=model.device,
    )
    return (
        interpolate_pos_encoding,
        model,
        query_inputs,
        ref_img_paths,
        target_frame,
        target_inputs,
    )


@app.cell
def _(
    forward,
    interpolate_pos_encoding,
    model,
    query_inputs,
    score_threshold_slider,
    target_frame,
    target_inputs,
):
    from torchvision.transforms.v2.functional import convert_bounding_box_format
    from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
    from transformers.image_utils import get_image_size


    def _():
        score_threshold = score_threshold_slider.value

        pred_boxes, scores = forward(
            model,
            query_inputs.query_pixel_values,
            target_inputs.pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        score_mask = scores.squeeze(-1) > score_threshold
        selected_boxes = pred_boxes[score_mask]
        selected_scores = scores[score_mask]

        pixel_boxes = BoundingBoxes(
            selected_boxes * max(canvas_size := get_image_size(target_frame)),
            format=BoundingBoxFormat.CXCYWH,
            canvas_size=canvas_size,
        )
        pixel_boxes = convert_bounding_box_format(
            pixel_boxes, new_format=BoundingBoxFormat.XYXY
        )
        return pixel_boxes, selected_scores


    pixel_boxes, selected_scores = _()
    return pixel_boxes, selected_scores


@app.cell
def _(mo, pixel_boxes, ref_img_paths, selected_scores, target_frame):
    from PIL import Image, ImageDraw
    from torchvision.transforms.functional import to_pil_image

    _vis_img = to_pil_image(target_frame)
    _draw = ImageDraw.Draw(_vis_img)
    for _box_tensor, _score in zip(pixel_boxes, selected_scores):
        _box = _box_tensor.tolist()
        _draw.text(_box[:2], str(_score.item()))
        _draw.rectangle(_box)

    _ref_tab = mo.carousel(mo.image(path, rounded=True) for path in ref_img_paths)
    _frame_tab = mo.image(_vis_img, caption="Input", rounded=True)

    mo.ui.tabs({"References": _ref_tab, "Result": _frame_tab})
    return


if __name__ == "__main__":
    app.run()
