from ultralytics import YOLO

model = YOLO('yolov12n.pt')

# Train the model
results = model.train(
  data='uav.yaml',
  epochs=300,
  patience=30,
  batch=48, 
  imgsz=960,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  seed = 42,
  cache = True,
  device="0",
  project='runs_UAV',
  name='yolov12n_bs48_img960'
)

# Evaluate model performance on the validation set
metrics = model.val()