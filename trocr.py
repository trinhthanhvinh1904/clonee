import albumentations as A
import score_processor
import string_processor
import numpy as np
import cv2

list_transform = [A.Rotate((-5, -5)), A.Rotate((5, 5))]

def read_score(source, processor, model, device):
    pixel_values = processor(source, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text, score_processor.process_string(text)


def read_studentId(source, processor, model, device, allow_list):
    pixel_values = processor(source, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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

            pixel_values = processor(aug_img, return_tensors="pt").pixel_values.to(
                device
            )
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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

        pixel_values = processor(binary_thresh_rgb, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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

                pixel_values = processor(binary_thresh_rgb, return_tensors="pt").pixel_values.to(
                    device
                )
                generated_ids = model.generate(pixel_values)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                best_match, max_score = string_processor.find_best_match(
                    ocr_result=text, list_ids=allow_list
                )

                if max_score > 0.35:
                    return text, best_match

    return text, "Error"


def read_subjectId(source, processor, model, device, allow_list):
    # img_cv2 = np.array(source)[:, :, ::-1].copy()
    gray_image = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    _, binary_thresh = cv2.threshold(gray_image, 160, 255, cv2.THRESH_BINARY)

    binary_thresh_rgb = cv2.merge([binary_thresh, binary_thresh, binary_thresh])

    pixel_values = processor(binary_thresh_rgb, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    best_match, max_score = string_processor.find_best_match(
        ocr_result=text, list_ids=allow_list
    )

    return text, best_match
