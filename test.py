import albumentations as A
import cv2
import xml.etree.ElementTree as ET
import glob


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

    for i in range(5):
        augmentations = transform(image=img, bboxes=bbox)
        augmented_img = augmentations["image"]
        images_list.append(augmented_img)
        saved_bboxes.append(list(augmentations["bboxes"][0]))
    return images_list,saved_bboxes


def write_image_box(images_list , bboxes_list, path, filename):
    for i in range(len(images_list)):
        cv2.imwrite(path+'images/'+filename+ str(i) + '.jpg', images_list[i])
        with open(path+'labels/'+filename+ str(i) +'.txt', 'w') as file:
            # Writing each element of the list to a new line in the file
            for num in bboxes_list[i]:
                file.write(str(num) + '\n')

img_paths = glob.glob("/home/abhiram/PycharmProjects/ASLDetection/ASL_Pascal_Voc/train/*.jpg")

i = 0

for img_path in img_paths:
    if i : break
    i += 1
    img, bboxes = read_img_box(img_path[:-3])
    images_list, saved_bboxes = augment_img(img, bboxes)
    # for j in range(len(images_list)):
    #     show_img_box(images_list[j],saved_bboxes[j])
    print(img_path[64:-4])
    save_path = "/home/abhiram/PycharmProjects/ASLDetection/AUG_ASL_Pascal_Voc/train/"
    write_image_box(images_list,saved_bboxes,save_path,img_path[64:-4])





# for img_path in img_paths:
#     img, bboxes = read_img_box(img_path[:-3])

# for img_path in img_paths:
#     box_path = img_path[:-3]+'xml'
#
#     image = cv2.imread(img_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     tree = ET.parse(box_path)
#     root = tree.getroot()
#     yolo_box = [[]]
#
#     for obj in root.findall('object'):
#         bbox = obj.find('bndbox')
#         xmin = int(bbox.find('xmin').text)
#         yolo_box[0].append(xmin)
#         ymin = int(bbox.find('ymin').text)
#         yolo_box[0].append(ymin)
#         xmax = int(bbox.find('xmax').text)
#         yolo_box[0].append(xmax)
#         ymax = int(bbox.find('ymax').text)
#         yolo_box[0].append(ymax)


    # transform = A.Compose(
    #     [
    #         A.Flip(p=1),
    #     ], bbox_params= A.BboxParams(format='pascal_voc',label_fields=[])
    # )
    #
    #
    # images_list = []
    # saved_bboxes = []
    #
    # for i in range(1):
    #     augmentations = transform(image = image, bboxes = yolo_box)
    #     augmented_img = augmentations["image"]
    #     images_list.append(augmented_img)
    #     saved_bboxes.append(list(augmentations["bboxes"][0]))

    # Displaying the image

    # for i in range(len(images_list)):
    #     bbox = saved_bboxes[i]
    #     xmin,ymin,xmax,ymax = bbox
    #     img = images_list[i]
    # #     # Draw a rectangle on the image
    #     print(bbox)
    #     cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
    # #     # Display the image with the bounding box
    #     cv2.imshow('Image with Bounding Box', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

