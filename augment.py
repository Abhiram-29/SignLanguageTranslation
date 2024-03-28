import albumentations as A
import cv2
import xml.etree.ElementTree as ET
import glob
import time


def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]


def read_img_box(path):
    img = cv2.imread(path+'jpg')
    tree = ET.parse(path+'xml')
    root = tree.getroot()
    bboxes = [[]]
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        bboxes[0].append(xmin)
        ymin = int(bbox.find('ymin').text)
        bboxes[0].append(ymin)
        xmax = int(bbox.find('xmax').text)
        bboxes[0].append(xmax)
        ymax = int(bbox.find('ymax').text)
        bboxes[0].append(ymax)
    return img,bboxes

def show_img_box(img, bbox):
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
    # Display the image with the bounding box
    cv2.imshow('Image with Bounding Box', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def augment_img(img,bbox):
    transform = A.Compose(
        [
            A.GaussianBlur(),
            A.BBoxSafeRandomCrop(),
            A.OneOf(
                [A.ChannelDropout(), A.ChannelShuffle(), A.ColorJitter()],
            ),
            A.OneOf(
                [A.ToSepia(), A.ToGray()],
            ),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.25),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[])
    )

    images_list = [img]
    saved_bboxes = [bbox[0]]
    try:
        for i in range(6):
            augmentations = transform(image=img, bboxes=bbox)
            augmented_img = augmentations["image"]
            images_list.append(augmented_img)
            saved_bboxes.append(list(augmentations["bboxes"][0]))
    except ValueError:
        print(f"Skipping invalid bounding box: {bbox}")
    return images_list,saved_bboxes


def write_image_box(images_list , bboxes_list, path, filename):
    for i in range(len(images_list)):
        cv2.imwrite(path+'images/'+filename+ str(i) + '.jpg', images_list[i])
        height, width, channels = images_list[i].shape
        x1,y1,x2,y2 = bboxes_list[i]
        index = ord(filename[0])-ord('A')
        yolo_bbox = [index]
        yolo_bbox += pascal_voc_to_yolo(x1,y1,x2,y2,width,height)
        with open(path+'labels/'+filename+ str(i) +'.txt', 'w') as file:
            # Writing each element of the list to a new line in the file
            for num in yolo_bbox:
                file.write(str(num) + ' ')


start_time = time.time()

img_paths = glob.glob("/home/abhiram/PycharmProjects/ASLDetection/ASL_Pascal_Voc/train/*.jpg")


for img_path in img_paths:
    img, bboxes = read_img_box(img_path[:-3])
    images_list, saved_bboxes = augment_img(img, bboxes)
    # for j in range(len(images_list)):
    #     show_img_box(images_list[j],saved_bboxes[j])
    save_path = "/home/abhiram/PycharmProjects/ASLDetection/AUG_ASL_Pascal_Voc/train/"
    write_image_box(images_list,saved_bboxes,save_path,img_path[64:-4])

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")