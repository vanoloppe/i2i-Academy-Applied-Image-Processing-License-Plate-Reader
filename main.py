import cv2
import numpy as np
import easyocr

image_path = "araba.jpg"  
img = cv2.imread(image_path)

if img is None:
    print(f"Hata: '{image_path}' dosyası bulunamadı! Lütfen yolu kontrol edin.")
    exit()

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# sınırlarını buluyoruz
edges = cv2.Canny(blurred, 30, 200)


contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

screen_cnt = None


for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    
    if len(approx) == 4:
        screen_cnt = approx
        break

if screen_cnt is None:
    print("Plaka konturu tespit edilemedi. Lütfen daha net bir görsel deneyin.")
    exit()

x, y, w, h = cv2.boundingRect(screen_cnt)

cropped_plate = img[y:y+h, x:x+w]


reader = easyocr.Reader(['tr', 'en'])

result = reader.readtext(cropped_plate)

if result:
    plate_text = result[0][1].strip().upper()
    print("\n" + "="*30)
    print(f"Tanınan Plaka: {plate_text}")
    print("="*30 + "\n")
else:
    print("Plaka bulundu ancak üzerindeki yazı okunamadı.")

cv2.imshow("Orijinal Görüntü", img)
cv2.imshow("Kırpılmış Plaka", cropped_plate)
cv2.waitKey(0)
cv2.destroyAllWindows()

