from ultralytics import YOLOv10
import torch

def main():
    student_ids = [
        "22BI13001", "22BI13007", "22BI13008", "22BI13009", "22BI13012", "22BI13013", "22BI13015", 
        "22BI13016", "22BI13018", "22BI13019", "22BI13021", "22BI13022", "22BI13023", "22BI13029", 
        "22BI13032", "22BI13034", "22BI13037", "22BI13042", "22BI13043", "22BI13045", "22BI13047", 
        "22BI13052", "22BI13055", "22BI13059", "22BI13068", "22BI13071", "22BI13077", "22BI13079", 
        "22BI13081", "22BI13085", "22BI13088", "22BI13089", "22BI13092", "22BI13093", "22BI13094", 
        "22BI13095", "22BI13096", "22BI13097", "22BI13098", "22BI13104", "22BI13106", "22BI13107", 
        "22BI13115", "22BI13116", "22BI13117", "22BI13118", "22BI13122", "22BI13123", "22BI13126", 
        "22BI13127", "22BI13128", "22BI13132", "22BI13136", "22BI13146", "22BI13147", "22BI13148", 
        "22BI13149", "22BI13150", "22BI13154", "22BI13158", "22BI13161", "22BI13163", "22BI13170", 
        "22BI13172", "22BI13181", "22BI13184", "22BI13186", "22BI13191", "22BI13201", "22BI13206", 
        "22BI13211", "22BI13222", "22BI13240", "22BI13242", "22BI13244", "22BI13259", "22BI13261", 
        "22BI13264", "22BI13274", "22BI13276", "22BI13278", "22BI13279", "22BI13282", "22BI13286", 
        "22BI13288", "22BI13291", "22BI13301", "22BI13307", "22BI13308", "22BI13317", "22BI13318", 
        "22BI13321", "22BI13323", "22BI13325", "22BI13328", "22BI13329", "22BI13331", "22BI13332", 
        "22BI13338", "22BI13340", "22BI13342", "22BI13343", "22BI13344", "22BI13346", "22BI13356", 
        "22BI13358", "22BI13367", "22BI13370", "22BI13371", "22BI13372", "22BI13374", "22BI13379", 
        "22BI13380", "22BI13386", "22BI13390", "22BI13392", "22BI13393", "22BI13394", "22BI13400", 
        "22BI13404", "22BI13406", "22BI13407", "22BI13410", "22BI13412", "22BI13419", "22BI13420", 
        "22BI13426", "22BI13434", "22BI13435", "22BI13436", "22BI13438", "22BI13443", "22BI13446", 
        "22BI13451", "22BI13453", "22BI13454", "22BI13463", "22BI13464", "22BI13474", "22BI13478", 
        "22BI13479", "22BI13480", "22BI13482", "22BI13485", "BA12-003", "BA12-006", "BA12-007", 
        "BA12-015", "BA12-016", "BA12-022", "BA12-034", "BA12-035", "BA12-045", "BA12-050", "BA12-057", 
        "BA12-062", "BA12-068", "BA12-078", "BA12-082", "BA12-084", "BA12-088", "BA12-089", "BA12-090", 
        "BA12-092", "BA12-093", "BA12-095", "BA12-102", "BA12-110", "BA12-126", "BA12-127", "BA12-128", 
        "BA12-134", "BA12-138", "BA12-139", "BA12-150", "BA12-180", "BA12-189", "BA12-192", "BA12-193", 
        "BI12-028", "BI12-079", "BI12-100", "BI12-123", "BI12-130", "BI12-144", "BI12-148", "BI12-183", 
        "BI12-197", "BI12-210", "BI12-217", "BI12-233", "BI12-251", "BI12-252", "BI12-277", "BI12-300", 
        "BI12-305", "BI12-336", "BI12-388", "BI11-183", "BI11-234"
    ]
    subject_ids = ["ICT3.002", "ICT2.001"]

    source_dir = "TrainImagesNew"
    yolo_path = 'weights/best.pt'

    predict(source_dir=source_dir, yolo_path=yolo_path, student_ids=student_ids, subject_ids=subject_ids)

def predict(
    source_dir=None,
    yolo_path=None,
    student_ids=None,
    subject_ids=None
):
    """
    Use YOLOv10 to detect text areas of interest and EasyOCR to recognize

    Parameters:
        source_dir: folder containing images that need to extract information. Make sure to put '/' at the end.
        Accept only PNG and JPG images

        yolo_path: path to YOLOv10 weight file.

        student_ids: word list for OCR result.

        subject_ids: word list for OCR result.
    """

    if (
        source_dir is None
        or yolo_path is None
        or student_ids is None
        or subject_ids is None
    ):
        print("Missing configuration! Aborting...")
        return

    """ setup """
    # import libraries
    import os
    from PIL import Image
    import cv2
    import time
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from trocr import read_score, read_studentId, read_subjectId
    from EasyOCR import read_score_easyocr, read_studentId_easyocr, read_subjectId_easyocr
    import easyocr

    # timer
    start = time.time()

    # device to run OCR model (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # models
    yolo_model = YOLOv10("weights/last.pt")

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    ocr_model = VisionEncoderDecoderModel.from_pretrained(
        'microsoft/trocr-large-handwritten' 
    ).to(device)

    gpu = torch.cuda.is_available()

    model_dir = 'models/easyocr/'
    easyocr_score = easyocr.Reader(['en'], gpu=gpu, model_storage_directory=model_dir)
    easyocr_studentId = easyocr.Reader(['en'], gpu=gpu, model_storage_directory=model_dir)
    easyocr_subjectId = easyocr.Reader(['en'], gpu=gpu, model_storage_directory=model_dir)

    # result file
    output_file = open("output_TrOCR.csv", "w")
    log_yolo = open("output_yolo.txt", "w")

    print('subjectId,studentId,score', file=output_file)
    # access each image
    for filename in os.listdir(source_dir):
        if not filename.endswith(".jpg") and not filename.endswith(".png"):
            continue

        img_path = os.path.join(source_dir, filename)
        img_cv2 = cv2.imread(img_path)
        predictions = yolo_model.predict(source=img_cv2)
        # filename_noext = os.path.splitext(filename)[0]

        # use YOLO model to predict and crop image
        score = None
        studentId = None
        subjectId = None
        for idx, prediction in enumerate(predictions):
            yolo_prediction_string = f"{filename}\n"
            
            for box_idx, box in enumerate(prediction.boxes):
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

                class_name = prediction.names[box.cls[0].item()]

                # get confidence score
                # confidence = box.conf[0].item()

                yolo_prediction_string += (
                    f"{box_idx}> {class_name}: ({xmin}, {ymin}, {xmax}, {ymax})\n"
                )

                # crop the region
                cropped_img = img_cv2[ymin:ymax, xmin:xmax]

                cropped_clone = cropped_img.copy()

                # read text from the cropped image
                if class_name == "Score":
                    # TrOCR
                    ocr_score, score = read_score(source=cropped_clone, processor=processor, model=ocr_model, device=device)
                    
                    # easyocr
                    # ocr_score, score = read_score_easyocr(cropped_clone, easyocr_score)
                elif class_name == "StudentID":
                    # TrOCR
                    ocr_studentId, studentId = read_studentId(source=cropped_clone, processor=processor, model=ocr_model, device=device, allow_list=student_ids)

                    # easyocr
                    # ocr_studentId, studentId = read_studentId_easyocr(cropped_clone, easyocr_studentId, student_ids)
                elif class_name == "SubjectID":
                    # TrOCR
                    ocr_subjectId, subjectId = read_subjectId(source=cropped_clone, processor=processor, model=ocr_model, device=device, allow_list=subject_ids)

                    # easyocr
                    # ocr_subjectId, subjectId = read_subjectId_easyocr(cropped_clone, easyocr_subjectId, subject_ids)
                else:
                    print("Unexpected class name detected!")

            print(yolo_prediction_string, file=log_yolo)

        # log the result
        if score is None:
            print('Score is not touched')
            score = 'Error'
            continue
        if studentId is None:
            print('StudentID is not touched')
            studentId = 'Error'
            continue
        if subjectId is None:
            print('SubjectID is not touched')
            subjectId = 'Error'
            continue

        print(f'{subjectId},{studentId},{score}', file=output_file)

    output_file.close()
    log_yolo.close()

    print("Finished!")
    end = time.time()
    elapsed_time = end - start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    log_time = open('elapsed_time_TrOCR.txt', 'w')
    print(f"Elapsed Time: {hours} hours, {minutes} minutes, {seconds} seconds")
    print(f"Elapsed Time: {hours} hours, {minutes} minutes, {seconds} seconds", file=log_time)
    log_time.close()

if __name__ == "__main__":
    main()