import pandas as pd
import os
import cv2
import yaml
import albumentations as A
from matplotlib import pyplot as plt

height = 2048
width = 2048

transform1 = A.Compose(
    [
        A.Rotate(limit=10),
        A.RandomBrightnessContrast(brightness_limit=0.2),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)

transform2 = A.Compose(
    [
        A.Rotate(limit=10),
        A.RandomBrightnessContrast(brightness_limit=0.2),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)

transform3 = A.Compose(
    [
        A.Rotate(limit=10),
        A.RandomBrightnessContrast(brightness_limit=0.2),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)

transform4 = A.Compose(
    [
        A.Rotate(limit=10),
        A.RandomBrightnessContrast(brightness_limit=0.2),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)

transforms = [transform1, transform2, transform3, transform4]

def augment(
    img_list,
    bbox_list,
    transforms,
    num_aug=4,
    cls_names=None,
    data_dir=None,
    og_bbox_value_list=None,
):
    # create directories
    output_dir = data_dir + "aug/"
    output_dir_full = output_dir + "full_img/"  # YOLOv10 dataset
    output_dir_cropped = output_dir + "cropped/"
    metadata_file_path = output_dir_full + "data.yaml"

    yolo_img_dir = output_dir_full + "images/"
    yolo_label_dir = output_dir_full + "labels/"

    full_img_train_dir = output_dir_full + "train/"
    full_img_val_dir = output_dir_full + "valid/"
    full_img_test_dir = output_dir_full + "test/"

    output_dir_cropped_subject_id = output_dir_cropped + "SubjectID/"
    output_dir_cropped_score = output_dir_cropped + "Score/"
    output_dir_cropped_student_id = output_dir_cropped + "StudentID/"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_full, exist_ok=True)
    os.makedirs(output_dir_cropped, exist_ok=True)

    os.makedirs(full_img_train_dir, exist_ok=True)
    os.makedirs(full_img_val_dir, exist_ok=True)
    os.makedirs(full_img_test_dir, exist_ok=True)

    os.makedirs(output_dir_cropped_subject_id, exist_ok=True)
    os.makedirs(output_dir_cropped_score, exist_ok=True)
    os.makedirs(output_dir_cropped_student_id, exist_ok=True)

    # read csv
    # path to original csv
    csv_subject_id_path = data_dir + "data_subjectId.csv"
    csv_score_path = data_dir + "data_score.csv"
    csv_student_id_path = data_dir + "data_studentId.csv"
    # path to new csv after augmentation
    out_csv_subject_id_path = output_dir_cropped_subject_id + "data_subjectId.csv"
    out_csv_score_path = output_dir_cropped_score + "data_score.csv"
    out_csv_student_id_path = output_dir_cropped_student_id + "data_studentId.csv"

    subject_ids = pd.read_csv(csv_subject_id_path)
    scores = pd.read_csv(csv_score_path)
    student_ids = pd.read_csv(csv_student_id_path)

    # data for easyocr
    augmented_data_subjectId = []
    augmented_data_score = []
    augmented_data_studentId = []

    # data for yolov10
    if len(img_list) != len(og_bbox_value_list):
        print(
            f"Image list and label group list mismatch! We have {len(img_list)} images and {len(og_bbox_value_list)} label groups"
        )
        return

    train = 0.9
    val = 0.05
    test = 1 - train - val
    counter = 1
    for i, img_item in enumerate(img_list):
        """split train, valid, test sets"""
        # print(f"counter: {counter}")
        if counter <= round(train * len(img_list)):
            print("train")
            yolo_img_dir = full_img_train_dir + "images/"
            yolo_label_dir = full_img_train_dir + "labels/"

            os.makedirs(yolo_img_dir, exist_ok=True)
            os.makedirs(yolo_label_dir, exist_ok=True)
        elif counter > round(train * len(img_list)) and counter <= round(
            (train + val) * len(img_list)
        ):
            print("valid")
            yolo_img_dir = full_img_val_dir + "images/"
            yolo_label_dir = full_img_val_dir + "labels/"

            os.makedirs(yolo_img_dir, exist_ok=True)
            os.makedirs(yolo_label_dir, exist_ok=True)
        else:
            print("test")
            yolo_img_dir = full_img_test_dir + "images/"
            yolo_label_dir = full_img_test_dir + "labels/"

            os.makedirs(yolo_img_dir, exist_ok=True)
            os.makedirs(yolo_label_dir, exist_ok=True)

        for filename, img in img_item.items():
            print(f"begin augmenting image {filename}")
            """ get original bbox group """
            bbox_group = og_bbox_value_list[i]

            # print(f"bbox_group of image {filename}: \n{bbox_group}")
            # for cls in bbox_group:
            #     print(f"{cls[0]} {cls[1]} {cls[2]} {cls[3]}")
            # break

            # {'Score': {}, 'StudentID': {}, 'SubjectID': {}}
            bboxes_of_img = bbox_list[i]
            # print(f"bboxes_of_img: {bboxes_of_img}\n")

            bbox_score = [
                bboxes_of_img[cls_names[0]]["x_min"],
                bboxes_of_img[cls_names[0]]["y_min"],
                bboxes_of_img[cls_names[0]]["x_max"],
                bboxes_of_img[cls_names[0]]["y_max"],
            ]
            bbox_studentId = [
                bboxes_of_img[cls_names[1]]["x_min"],
                bboxes_of_img[cls_names[1]]["y_min"],
                bboxes_of_img[cls_names[1]]["x_max"],
                bboxes_of_img[cls_names[1]]["y_max"],
            ]
            bbox_subjectId = [
                bboxes_of_img[cls_names[2]]["x_min"],
                bboxes_of_img[cls_names[2]]["y_min"],
                bboxes_of_img[cls_names[2]]["x_max"],
                bboxes_of_img[cls_names[2]]["y_max"],
            ]

            bboxes_transform = [bbox_score, bbox_studentId, bbox_subjectId]

            """ export original image and labels to the new dataset """
            name = f"{i+1}"
            new_filename_original = f"{name}.jpg"
            new_label_filename_original = f"{name}.txt"
            original_img_path = os.path.join(yolo_img_dir, new_filename_original)
            original_label_path = os.path.join(
                yolo_label_dir, new_label_filename_original
            )
            cv2.imwrite(original_img_path, img)
            label_file = open(original_label_path, "w")
            for cls in bbox_group:
                label_file.write(f"{cls[0]} {cls[1]} {cls[2]} {cls[3]} {cls[4]}\n")
            label_file.close()

            # export original image's csv data
            subjectId_row = subject_ids[
                subject_ids["filename"] == new_filename_original
            ]
            score_row = scores[scores["filename"] == new_filename_original]
            studentId_row = student_ids[
                student_ids["filename"] == new_filename_original
            ]
            og_subjectId = subjectId_row.iloc[0]["subjectId"]
            og_score = score_row.iloc[0]["score"]
            og_studentId = studentId_row.iloc[0]["studentId"]

            # print(
            #     f"subjectId_row = {og_subjectId}\nscore_row={og_score}\nstudentId_row={og_studentId}"
            # )

            """ export original image's cropped parts """
            cropped_name_subjectId = f"{i+1}_SubjectID"
            cropped_name_score = f"{i+1}_Score"
            cropped_name_studentId = f"{i+1}_StudentID"

            original_img_subjectId = f"{cropped_name_subjectId}.jpg"
            original_img_score = f"{cropped_name_score}.jpg"
            original_img_studentId = f"{cropped_name_studentId}.jpg"

            og_subjectId_path = os.path.join(
                output_dir_cropped_subject_id, original_img_subjectId
            )
            og_score_path = os.path.join(output_dir_cropped_score, original_img_score)
            og_studentId_path = os.path.join(
                output_dir_cropped_student_id, original_img_studentId
            )
            part_subjectId = img[
                bbox_subjectId[1] : bbox_subjectId[3],
                bbox_subjectId[0] : bbox_subjectId[2],
            ]
            part_score = img[
                bbox_score[1] : bbox_score[3], bbox_score[0] : bbox_score[2]
            ]
            part_studentId = img[
                bbox_studentId[1] : bbox_studentId[3],
                bbox_studentId[0] : bbox_studentId[2],
            ]
            cv2.imwrite(og_subjectId_path, part_subjectId)
            cv2.imwrite(og_score_path, part_score)
            cv2.imwrite(og_studentId_path, part_studentId)

            augmented_data_subjectId.append(
                {"filename": original_img_subjectId, "subjectId": og_subjectId}
            )
            augmented_data_score.append(
                {"filename": original_img_score, "score": og_score}
            )
            augmented_data_studentId.append(
                {"filename": original_img_studentId, "studentId": og_studentId}
            )

            """ apply transformation """
            for h, transform in enumerate(transforms):
                for j in range(num_aug):
                    while True:
                        # print(f"bboxes_transform: {bboxes_transform}")
                        augmented = transform(
                            image=img, bboxes=bboxes_transform, labels=cls_names
                        )
                        augmented_img = augmented["image"]
                        augmented_bboxes = augmented["bboxes"]
                        # print(f"how many boxes: {len(augmented_bboxes)}")
                        if len(augmented_bboxes) != 3:
                            continue
                        else:
                            break

                    """ save augmented full image and labels """
                    new_filename = f"{i+1}_aug_{h+1}_v{j+1}"
                    new_filename_full = f"{new_filename}.jpg"
                    new_label_filename = f"{new_filename}.txt"
                    augmented_full_img_path = os.path.join(
                        yolo_img_dir, new_filename_full
                    )
                    cv2.imwrite(augmented_full_img_path, augmented_img)

                    augmented_label_path = os.path.join(
                        yolo_label_dir, new_label_filename
                    )
                    augmented_img_label_file = open(augmented_label_path, "w")
                    for cls_idx, pascal_voc_values in enumerate(augmented_bboxes):
                        aug_x_min, aug_y_min, aug_x_max, aug_y_max = pascal_voc_values
                        # print(f"{cls_idx} {aug_x_min} {aug_y_min} {aug_x_max} {aug_y_max}\n")

                        """ convert back to roboflow format """
                        bbox_width = aug_x_max - aug_x_min
                        bbox_height = aug_y_max - aug_y_min
                        x_center = aug_x_min + bbox_width / 2
                        y_center = aug_y_min + bbox_height / 2

                        x_center_ratio = x_center / width
                        y_center_ratio = y_center / height
                        bbox_width_ratio = bbox_width / width
                        bbox_height_ratio = bbox_height / height

                        augmented_img_label_file.write(
                            f"{cls_idx} {x_center_ratio} {y_center_ratio} {bbox_width_ratio} {bbox_height_ratio}\n"
                        )

                    augmented_img_label_file.close()

                    # crop areas of interest
                    for class_id, bbox in enumerate(augmented_bboxes):
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        cropped_img = augmented_img[y_min:y_max, x_min:x_max]
                        if class_id not in range(3):
                            cls_name = "new"
                        else:
                            cls_name = cls_names[class_id]

                        new_filename_cropped = f"{i+1}_aug_{h+1}_v{j+1}-{cls_name}.jpg"
                        cropped_augmented_img_path = os.path.join(
                            (
                                output_dir_cropped_subject_id
                                if class_id == 2
                                else (
                                    output_dir_cropped_score
                                    if class_id == 0
                                    else output_dir_cropped_student_id
                                )
                            ),
                            new_filename_cropped,
                        )
                        cv2.imwrite(cropped_augmented_img_path, cropped_img)

                        # export csv data for new cropped part
                        if cls_name == "SubjectID":
                            augmented_data_subjectId.append(
                                {
                                    "filename": new_filename_cropped,
                                    "subjectId": og_subjectId,
                                }
                            )
                        elif cls_name == "Score":
                            augmented_data_score.append(
                                {"filename": new_filename_cropped, "score": og_score}
                            )
                        else:
                            augmented_data_studentId.append(
                                {
                                    "filename": new_filename_cropped,
                                    "studentId": og_studentId,
                                }
                            )
        counter += 1

    nb_aug_full_imgs = len(os.listdir(yolo_img_dir))  # number of augmented full images
    nb_aug_cropped_imgs = (
        len(os.listdir(output_dir_cropped_subject_id))
        + len(os.listdir(output_dir_cropped_score))
        + len(os.listdir(output_dir_cropped_student_id))
    )  # number of augmented cropped images
    print(
        f"After augmentation, obtained {nb_aug_full_imgs} full images, {nb_aug_cropped_imgs} cropped images"
    )

    df_augmented_data_subjectId = pd.DataFrame(augmented_data_subjectId)
    df_augmented_data_score = pd.DataFrame(augmented_data_score)
    df_augmented_data_studentId = pd.DataFrame(augmented_data_studentId)

    df_augmented_data_subjectId.to_csv(out_csv_subject_id_path, index=False)
    df_augmented_data_score.to_csv(out_csv_score_path, index=False)
    df_augmented_data_studentId.to_csv(out_csv_student_id_path, index=False)

    """ generate data.yaml file for the new dataset """
    with open(f"{data_dir}data.yaml", "r") as og_data_file:
        og_data = og_data_file.read()
    with open(metadata_file_path, "w") as metadata_file:
        metadata_file.write(og_data)
    metadata_file.close()
    print(f"created data.yaml at {metadata_file_path}")

    return metadata_file_path


# def augment_subjectId(transforms):
#     img_dir = "datasets/SubjectID/train/images/"
#     img_path = img_dir + "20241128_193451_jpg.rf.bb2c5a52204b546e469d99b48ce15264.jpg"
#     label_dir = "datasets/SubjectID/train/labels/"
#     label_path = (
#         label_dir + "20241128_193451_jpg.rf.bb2c5a52204b546e469d99b48ce15264.txt"
#     )
#     data_path = 'datasets/SubjectID/data.yaml'
#     # read the class names given by roboflow
#     metadata = read_yaml(data_path)
#     cls_names = metadata.get("names", [])

#     img_list = []
#     bbox_list = []
#     og_bbox_value_list = []
#     with open(label_path, "r") as label_file:
#         lines = label_file.readlines()
#         og_label_group = []
#         bboxes = {}
#         for i, line in enumerate(lines):
#             (
#                 class_id,
#                 x_center_ratio,
#                 y_center_ratio,
#                 bbox_width_ratio,
#                 bbox_height_ratio,
#             ) = line.strip().split()

#             og_label_group.append(
#                 [
#                     class_id,
#                     x_center_ratio,
#                     y_center_ratio,
#                     bbox_width_ratio,
#                     bbox_height_ratio,
#                 ]
#             )

#             x_center = float(x_center_ratio) * width
#             y_center = float(y_center_ratio) * height
#             bbox_width = float(bbox_width_ratio) * width
#             bbox_height = float(bbox_height_ratio) * height

#             """ convert to pascal_voc format for augmentation"""
#             x_min = int(x_center - (bbox_width / 2))
#             y_min = int(y_center - (bbox_height / 2))
#             x_max = int(x_center + (bbox_width / 2))
#             y_max = int(y_center + (bbox_height / 2))

#             label = cls_names[int(class_id)]
#             # print(f'class name: {label}')

#             bboxes[f'{label}_{i}'] = {
#                 "x_min": x_min,
#                 "y_min": y_min,
#                 "x_max": x_max,
#                 "y_max": y_max,
#             }

#             bbox_list.append(bboxes)
#             og_bbox_value_list.append(og_label_group)

#     print()
#     # augment
    
def augment_subjectId():
  print('begin augmenting SubjectID images')
  img_dir = '../tesstrain/data/trainset/SubjectID/'
  output_dir = img_dir + 'new/'
  os.makedirs(output_dir, exist_ok=True)
  img_list = []
  for filename in os.listdir(img_dir):
    if not filename.endswith('.tif'):
        continue
    
    img_path = os.path.join(img_dir, filename)
    gt_data = open(os.path.splitext(img_path)[0] + '.gt.txt').readline()
    img = cv2.imread(os.path.join(img_dir, filename))

    h, w, _ = img.shape

    x_min = 0
    y_min = 0
    x_max = w
    y_max = h

    bbox_subjectId = [x_min, y_min, x_max, y_max]

    for i, transform in enumerate(transforms):
        for j in range(4):
          augmented = transform(
              image=img, bboxes=[bbox_subjectId], labels=['SubjectID']
          )
          augmented_img = augmented["image"]
          augmented_bboxes = augmented["bboxes"]
          # print(f"how many boxes: {len(augmented_bboxes)}")

          new_filename = os.path.splitext(filename)[0] + f'_{i+1}_v{j}'
          new_img_path = os.path.join(output_dir, new_filename + '.tif')
          new_gt_path = os.path.join(output_dir, new_filename + '.gt.txt')

          cv2.imwrite(new_img_path, augmented_img)

          gt_file = open(new_gt_path, 'w')
          gt_file.write(gt_data)
          gt_file.close()
  print('finished')
        
            

      

def read_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data