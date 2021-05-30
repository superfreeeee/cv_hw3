import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from SVM import get_model

img_base = 'data/images/'
annotation_base = 'data/annotations_trainval/'
extract_base = 'data/extract_images_val/'
data_base = 'data/'

tmp_base = 'data/tmp/'
result_base = 'data/result/'
val_base = 'data/val_result/'


# origin grabcut test
# img = cv2.imread('data/images/0.jpg')
#
# img = get_img('data/images/0.jpg')
#
# mask = np.zeros(img.shape[:2], np.uint8)
#
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
#
# rect = (1, 1, img.shape[1], img.shape[0])
#
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
#
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#
# img2 = img * mask2[:, :, np.newaxis]
#
# plt.subplot(121)
# plt.imshow(img2)
# plt.title('grabcut')
# plt.xticks([])
# plt.yticks([])
#
# plt.subplot(122)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('original')
# plt.xticks([])
# plt.yticks([])
#
# plt.show()

def write_file(file, content):
    with open(file, 'a') as f:
        f.write(content)


def save_img(base, num, data, ext='.png'):
    cv2.imwrite('{}{}{}'.format(base, num, ext), data)


def save_result(base, num, data, ext='.png'):
    img = Image.fromarray(np.asarray(data, dtype='uint8'))
    # img.convert('L')
    img.save('{}{}{}'.format(base, num, ext))


# def save_extract_image(num, img):
#     cv2.imwrite('{}{}.png'.format(extract_base, num), img)


# def save_result(num, img):
#     cv2.imwrite('{}{}.png'.format(result_base, num), img)


def get_annotation(num):
    return np.asarray(Image.open('{}{}.png'.format(annotation_base, num)), dtype=np.uint8)


def get_img(num):
    return np.asarray(Image.open('{}{}.jpg'.format(img_base, num)), dtype=np.uint8)


def get_numbers(name):
    file = '{}{}.txt'.format(data_base, name)
    with open(file, 'r') as f:
        return f.read().split()


# abandon
# exclude_tags = {
#     '0,0,0',
#     '192,224,224'
# }
#
#
# def get_tags(annotation):
#     tags = set()
#     for row in range(annotation.shape[0]):
#         for col in range(annotation.shape[1]):
#             rgb = annotation[row][col]
#             tag = '{},{},{}'.format(rgb[0], rgb[1], rgb[2])
#             if tag not in exclude_tags and tag not in tags:
#                 tags.add(tag)
#     return list(tags)


# 图中所有颜色
def get_id_list(anno):
    ids = set()
    rows, cols = anno.shape
    for i in range(rows):
        for j in range(cols):
            id = anno[i][j]
            if id != 0 and id != 255:
                ids.add(id)
    return sorted(list(ids), key=int)


empty_color = np.array([0, 0, 0])


# 抽取目标颜色对应原图
def extract_img(img, anno, target_id):
    new_img = np.zeros(img.shape)
    rows, cols = anno.shape
    for i in range(rows):
        for j in range(cols):
            id = anno[i][j]
            new_img[i][j] = img[i][j] if id == target_id else empty_color
    return new_img


# abandon
# def filter_img(img, anno):
#     new_img = np.zeros(img.shape)
#     rows, cols = anno.shape
#     for i in range(rows):
#         for j in range(cols):
#             id = anno[i][j]
#             new_img[i][j] = img[i][j] if id != 0 and id != 255 else np.array([0, 0, 0])
#     return new_img


# abandon
# def build_black(img):
#     new_img = np.zeros(img.shape[:2])
#     return new_img


# abandon
# def get_tag_id(img, anno, id_to_tag):
#     rows, cols = anno.shape
#     for i in range(rows):
#         for j in range(cols):
#             id = anno[i][j]
#             if id != 0 and id != 255 and id not in id_to_tag:
#                 r, g, b = img[i][j]
#                 tag = '{},{},{}'.format(r, g, b)
#                 id_to_tag[id] = tag


# abandon
# def process_extract_images_id():
#     tag_to_id = {}
#
#     with open('data/id_to_tag.txt', 'r') as f:
#         lines = f.read().split('\n')[:-1]
#         for line in lines:
#             id, tag = line.split()
#             tag_to_id[tag] = id
#
#     print(tag_to_id)
#
#     number_to_id = []
#
#     with open('data/extract_number_to_tag.txt', 'r') as f:
#         lines = f.read().split('\n')[:-1]
#         for line in lines:
#             num, tag = line.split()
#             id = tag_to_id[tag] if tag in tag_to_id else 0
#             number_to_id.append('{} {} {}'.format(num, tag, id))
#
#     with open('data/extract_number_to_tag2.txt', 'w') as f:
#         for file in number_to_id:
#             f.write(file + '\n')
# process_extract_images_id()


# tag 转 id
# abandon
# def transform_tag_to_id():
#     nums = get_numbers('train')
#     print(len(nums))
#
#     id_to_tag = {}
#
#     for num in tqdm(nums):
#         img = get_img(num)
#         anno = get_annotation(num)
#         get_tag_id(img, anno, id_to_tag)
#
#     with open('data/id_to_tag.txt', 'w') as f:
#         for id, tag in id_to_tag.items():
#             f.write('{} {}\n'.format(id, tag))


# transform_tag_to_id()

# 抽取训练原图
def extract_items_from_images(target, tag_file):
    nums = get_numbers(target)
    print(len(nums))

    count = 0

    for num in tqdm(nums):
        img = get_img(num)
        anno = get_annotation(num)
        id_list = get_id_list(anno)
        for id in id_list:
            img2 = extract_img(img, anno, id)
            save_img(extract_base, count, img2)
            # save_extract_image(count, img2)
            write_file(tag_file, '{} {}\n'.format(count, id))
            count += 1


# extract_items_from_images('train', 'data/extract_number_to_tag3.txt')
# extract_items_from_images('val', 'data/extract_number_to_tag_val.txt')

# abandon
# tmp test
# for number in tqdm(nums):
#     img = get_img(number)
#     annotation = get_annotation(number)
#     if img is not None and annotation is not None:
#         tags = get_tags(annotation)
#         for tag in tags:
#             extract_img = filter_img(img, annotation, tag)
#             save_extract_image(str(counts), extract_img)
#
#             write_file('data/extract_number_to_tag.txt', '{} {}\n'.format(counts, tag))
#
#             counts += 1
# img2 = grab_cut(img)
# save_img(tmp_base, number, img2)

# save_extract_image(num, extract_img)
# tmp 2
# id_list = set()
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         id_list.add(img[i][j])
# print(list(id_list))

def fill_color(img, tag):
    new_img = np.zeros(img.shape[:2])
    rows, cols = img.shape[:2]
    for i in range(rows):
        for j in range(cols):
            r, g, b = img[i][j]
            new_img[i][j] = 0 if r == 0 and g == 0 and b == 0 else tag
    return new_img


def grab_cut(img):
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (1, 1, img.shape[1], img.shape[0])

    cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    return img * mask2[:, :, np.newaxis]


def classify_and_fill_color(model, mod, base):
    nums = get_numbers(mod)

    print('nums', len(nums))

    for num in tqdm(nums):
        img = get_img(num)

        if img is not None:
            img2 = grab_cut(img)

            img3 = Image.fromarray(img2).resize([100, 100], Image.ANTIALIAS)
            img4 = np.asarray(img3, dtype=np.int32).flatten()

            result = int(model.predict([img4])[0])
            img5 = fill_color(img2, result)

            save_result(result_base, num, img5)


if __name__ == '__main__':
    model = get_model()

    # test.txt
    # classify_and_fill_color(model, 'test', result_base)
    # val.txt
    classify_and_fill_color(model, 'val', val_base)

    pass

