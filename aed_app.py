import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# 初始化 MediaPipe 人體骨架偵測
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- 核心邏輯區 ---

def enhance_image(image):
    """影像增強：自動調整亮度與對比 (CLAHE)"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def detect_pads_color_based(image):
    """
    模擬 AED 貼片偵測 (基於顏色閾值)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 定義白/灰/銀色的範圍 (AED 貼片特徵)
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 50, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 形態學處理，去除雜訊
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_pads = []
    img_h, img_w = image.shape[:2]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 過濾太小(雜訊)或太大(背景)的區域
        if area > (img_h * img_w * 0.02) and area < (img_h * img_w * 0.3):
            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w//2, y + h//2)
            detected_pads.append({'rect': (x, y, w, h), 'center': center, 'type': 'unknown'})
    
    # 簡單區分左右貼片 (根據畫面位置)
    pads_sorted = sorted(detected_pads, key=lambda p: p['center'][0])
    final_pads = {}
    
    if len(pads_sorted) >= 1:
        for p in pads_sorted:
            cx, cy = p['center']
            # 畫面左半邊為 Sternum，右半邊為 Apex
            if cx < img_w / 2 and cy < img_h * 0.6:
                final_pads['sternum'] = p
            elif cx > img_w / 3 and cy > img_h * 0.4:
                final_pads['apex'] = p
                
    return final_pads

def analyze_placement(image, pads):
    """
    進行中間標準 (Intermediate Standard) 分析
    """
    results = pose.process(image)
    h, w, _ = image.shape
    
    feedback = []
    score = 100
    
    if not results.pose_landmarks:
        return image, ["無法偵測到人體，請調整拍攝角度"], 0

    landmarks = results.pose_landmarks.landmark
    
    # 關鍵點座標轉換
    right_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
    left_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
    right_hip = (int(landmarks[23].x * w), int(landmarks[23].y * h))
    left_hip = (int(landmarks[24].x * w), int(landmarks[24].y * h))

    annotated_img = image.copy()
    
    # 畫出參考線
    cv2.line(annotated_img, left_shoulder, left_hip, (255, 255, 0), 2)

    # --- 1. 右上貼片 (Sternum) ---
    if 'sternum' in pads:
        pad = pads['sternum']
        px, py, pw, ph = pad['rect']
        cx, cy = pad['center']
        
        cv2.rectangle(annotated_img, (px, py), (px+pw, py+ph), (0, 255, 0), 2)
        
        if cy > right_shoulder[1]: 
            feedback.append("✅ 右上貼片：位置正確")
        else:
            feedback.append("⚠️ 右上貼片：位置過高 (壓到鎖骨)")
            score -= 10
            cv2.rectangle(annotated_img, (px, py), (px+pw, py+ph), (0, 165, 255), 2)
    else:
        feedback.append("❓ 右上貼片：未偵測到")

    # --- 2. 左下貼片 (Apex) ---
    if 'apex' in pads:
        pad = pads['apex']
        px, py, pw, ph = pad['rect']
        cx, cy = pad['center']
        
        body_width_at_pad = left_hip[0] - right_hip[0]
        limit_line_x = left_shoulder[0] - (body_width_at_pad * 0.2)
        
        if cx > limit_line_x: 
            feedback.append("✅ 左下貼片：位置合格 (符合中間標準)")
            cv2.rectangle(annotated_img, (px, py), (px+pw, py+ph), (0, 255, 0), 2)
        elif cy > left_hip[1]: 
             feedback.append("❌ 左下貼片：位置錯誤 (貼在腹部)")
             score -= 50
             cv2.rectangle(annotated_img, (px, py), (px+pw, py+ph), (255, 0, 0), 3)
        else:
             feedback.append("⚠️ 左下貼片：稍嫌靠前 (建議往腋下移動)")
             score -= 20
             cv2.rectangle(annotated_img, (px, py), (px+pw, py+ph), (0, 255, 255), 2)
             
    else:
        feedback.append("❓ 左下貼片：未偵測到")

    return annotated_img, feedback, score

# --- Streamlit 介面 ---
st.set_page_config(page_title="AED 貼片位置檢核系統", page_icon="⚡")

st.title("⚡ AED 貼片位置智慧檢核")
st.markdown("""
**設計者**：禎的大腦 (新北市消防局高級救護技術員)
**用途**：透過 AI 影像辨識，分析 AED 貼片黏貼位置是否符合 AHA 指引。
**當前模式**：🟢 中間標準 (實務教學模式)
""")

uploaded_file = st.file_uploader("請上傳 AED 訓練或現場照片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始影像")
        st.image(image, use_column_width=True)
    
    with st.spinner('正在分析解剖位置與電流向量...'):
        enhanced_img = enhance_image(img_array)
        pads = detect_pads_color_based(enhanced_img)
        result_img, feedback_text, final_score = analyze_placement(enhanced_img, pads)
        
        with col2:
            st.subheader("分析結果")
            st.image(result_img, use_column_width=True)
            
        st.divider()
        st.header(f"整體評分：{final_score} 分")
        
        for item in feedback_text:
            if "❌" in item:
                st.error(item)
            elif "⚠️" in item:
                st.warning(item)
            else:
                st.success(item)
        
        if final_score < 80:
            st.info("💡 教官建議：左下側貼片請務必確認『腋中線』位置，避免貼於腹部軟組織。")