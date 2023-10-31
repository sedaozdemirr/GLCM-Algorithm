import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def glcmAlgorithm(img):
# Görüntüyü yükleyin ve gri seviye dönüştürün

    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu seviyesi ile görüntüyü siyah beyaz hale getirin
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # İlgi alanını ayıklayın (beyaz pikselleri seçin)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # İlgi alanının çevresini çizin
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    # Cilt bölgesini belirleyin
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)

    # GLCM oluşturun ve doku özelliklerini hesaplayın
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0][0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0][0]
    correlation = graycoprops(glcm, 'correlation')[0][0]
    energy = graycoprops(glcm, 'energy')[0][0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0][0]

    if   contrast >150  and dissimilarity < 10 and homogeneity < 0.5 and  energy < 0.5:
        print("Görüntüde akne tespit edildi.")
    else:
        print("Görüntüde akne tespit edilemedi.")

    # Sonuçları yazdırın
    print("Contrast:", contrast)
    print("Correlation:", correlation)
    print("Energy:", energy)
    print("Homogeneity:", homogeneity)
    print("diss:", dissimilarity)

    # Görüntüyü gösterin
    cv2.imshow("Skin", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

glcmAlgorithm('acne-Closed-Comedo.jpg')