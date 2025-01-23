import cv2
import supervision as sv
from supervision.utils.conversion import ensure_cv2_image_for_annotation
from supervision.annotators.utils import (resolve_color, resolve_text_background_xyxy)
from supervision.config import CLASS_NAME_DATA_FIELD

class MultilineLabelAnnotator(sv.LabelAnnotator):
    def __init__(self, line_separator='\n', **kwargs):
        super().__init__(**kwargs)
        self.line_separator = line_separator

    @ensure_cv2_image_for_annotation
    def annotate(self, scene, detections, labels=None, custom_color_lookup=None):
        font = cv2.FONT_HERSHEY_SIMPLEX
        anchors_coordinates = detections.get_anchors_coordinates(
            anchor=self.text_anchor
        ).astype(int)
        if labels is not None and len(labels) != len(detections):
            raise ValueError(
                f"The number of labels ({len(labels)}) does not match the "
                f"number of detections ({len(detections)}). Each detection "
                f"should have exactly 1 label."
            )

        for detection_idx, center_coordinates in enumerate(anchors_coordinates):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=(
                    self.color_lookup
                    if custom_color_lookup is None
                    else custom_color_lookup
                ),
            )

            if labels is not None:
                text = labels[detection_idx]
            elif detections[CLASS_NAME_DATA_FIELD] is not None:
                text = detections[CLASS_NAME_DATA_FIELD][detection_idx]
            elif detections.class_id is not None:
                text = str(detections.class_id[detection_idx])
            else:
                text = str(detection_idx)

            lines = text.split(self.line_separator)

            id_text_w, id_text_h = cv2.getTextSize(
                text=lines[0], fontFace=font, fontScale=self.text_scale, thickness=self.text_thickness
            )[0]
            id_text_w_padded = id_text_w + 2 * self.text_padding
            id_text_h_padded = id_text_h + 2 * self.text_padding

            box_center_x = int((detections.xyxy[detection_idx][0] + detections.xyxy[detection_idx][2]) // 2)
            box_center_y = detections.xyxy[detection_idx][3] + 10
            id_box_center_coordinates = (
                box_center_x,
                box_center_y
            )

            id_text_background_xyxy = tuple(map(int, resolve_text_background_xyxy(
                center_coordinates=tuple(id_box_center_coordinates),
                text_wh=(id_text_w_padded, id_text_h_padded),
                position=sv.Position.BOTTOM_CENTER,
            )))

            self.draw_rounded_rectangle(
                scene=scene,
                xyxy=id_text_background_xyxy,
                color=color.as_bgr(),
                border_radius=self.border_radius,
            )

            id_text_x = id_text_background_xyxy[0] + (id_text_w_padded - id_text_w) // 2
            id_text_y = id_text_background_xyxy[1] + id_text_h_padded - self.text_padding - id_text_h // 2

            cv2.putText(
                img=scene,
                text=lines[0],
                org=(id_text_x, id_text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )

            if len(lines) > 1:
                speed_text_w, speed_text_h = cv2.getTextSize(
                    text=lines[1], fontFace=font, fontScale=self.text_scale, thickness=self.text_thickness
                )[0]
                speed_text_x = box_center_x - speed_text_w // 2
                speed_text_y = id_text_background_xyxy[3] + 20  # 20 là khoảng cách giữa box ID và speed

                cv2.putText(
                    img=scene,
                    text=lines[1],
                    org=(speed_text_x, speed_text_y),
                    fontFace=font,
                    fontScale=self.text_scale,
                    color=self.text_color.as_rgb(),
                    thickness=self.text_thickness,
                    lineType=cv2.LINE_AA,
                )

        return scene

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)

triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=20, height=17,
    outline_thickness=1
)

label_annotator = MultilineLabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    border_radius=5,
    text_thickness=1,
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)

def annotate_human(frame, detections):
    annotated_frame = frame.copy()

    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    return annotated_frame

def annotate_ball(frame, detections):
    annotated_frame = frame.copy()

    annotated_frame = triangle_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    return annotated_frame

def annotate_label(frame, detections, labels):
    annotated_frame = frame.copy()

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    return annotated_frame