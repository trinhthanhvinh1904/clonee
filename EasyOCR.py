import albumentations as A
import easyocr.recognition
import score_processor
import string_processor
import numpy as np
import cv2
import easyocr

list_transform = [A.Rotate((-5, -5)), A.Rotate((5, 5))]

def read_score_easyocr(source, model):
    text = model.recognize(source, allowlist='0123456789.')[0][1]

    return text, score_processor.process_string(text)


def read_studentId_easyocr(source, model, allow_list):
    text = model.recognize(source, allowlist='0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ')[0][1]

    best_match, max_score = string_processor.find_best_match(
        ocr_result=text, list_ids=allow_list
    )

    if max_score > 0.35:
        return text, best_match
    else:
        print("Difficult to recognize, try rotating...")
        for i in range(len(list_transform)):
            transform = list_transform[i]

            # augment the source image
            augmented_result = transform(image=source)
            aug_img = augmented_result["image"]

            text = model.recognize(source, allowlist='0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ')[0][1]

            best_match, max_score = string_processor.find_best_match(
                ocr_result=text, list_ids=allow_list
            )

            if max_score > 0.35:
                return text, best_match
            
        # if score is still low, try thresholding
        print("Difficult to recognize, try thresholding...")
        gray_image = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        _, binary_thresh = cv2.threshold(gray_image, 160, 255, cv2.THRESH_BINARY)

        binary_thresh_rgb = cv2.merge([binary_thresh, binary_thresh, binary_thresh])

        text = model.recognize(source, allowlist='0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ')[0][1]

        best_match, max_score = string_processor.find_best_match(
            ocr_result=text, list_ids=allow_list
        )

        if max_score > 0.35:
            return text, best_match
        else:
            print("Difficult to recognize, try rotating and thresholding...")
            for i in range(len(list_transform)):
                transform = list_transform[i]

                # augment the source image
                augmented_result = transform(image=source)
                aug_img = augmented_result["image"]

                gray_image = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
                _, binary_thresh = cv2.threshold(gray_image, 160, 255, cv2.THRESH_BINARY)

                binary_thresh_rgb = cv2.merge([binary_thresh, binary_thresh, binary_thresh])

                text = model.recognize(source, allowlist='0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ')[0][1]

                best_match, max_score = string_processor.find_best_match(
                    ocr_result=text, list_ids=allow_list
                )

                if max_score > 0.35:
                    return text, best_match

    return text, "Error"


def read_subjectId_easyocr(source, model, allow_list):
    # img_cv2 = np.array(source)[:, :, ::-1].copy()
    gray_image = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    _, binary_thresh = cv2.threshold(gray_image, 160, 255, cv2.THRESH_BINARY)

    binary_thresh_rgb = cv2.merge([binary_thresh, binary_thresh, binary_thresh])

    text = model.recognize(source, allowlist='0123456789-.ABCDEFGHIJKLMNOPQRSTUVWXYZ')[0][1]

    best_match, max_score = string_processor.find_best_match(
        ocr_result=text, list_ids=allow_list
    )

    return text, best_match
