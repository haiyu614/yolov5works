<h2>训练过程</h3>
<p>数据集在data的demo目录下，分为images和labels两个文件夹，images文件夹存放训练图片，labels文件夹存放训练图片对应的标签文件。</p>
<p>
分别配置了两个yaml文件，一个是data的配置文件my_data.yaml，另一个是weigth的配置文件my_model.yaml。分别在data目录和models目录下创建。
模型选的是yolov5s，训练了100个epoch，batch-size为4，使用cpu进行训练。
</p>
<p>训练了3类，分别是cat、xuebao（雪豹）、dinosaur,训练了大概26张图片,训练过程中产生了一个警告：
<em>“FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.”<em></p>
<h3>训练命令：</h3>

```bash
  python train.py --data my_data.yaml --cfg my_model.yaml --weights yolov5s.pt --epoch 100 --batch-size 4 --device cpu 
```

<p>训练结果保存在了runs\train\exp4</p>

<h3>尝试用训练的模型检测测试图片</h3>

```bash
python detect.py --source D:/RCSlearn/yolov5_github/data/images/test --weights  D:/RCSlearn/yolov5_github/runs/train/exp4/weights/best.pt --conf 0.25 --imgsz 640 --device cpu
```

<p>runs\detect\exp3和exp4目录下分别保存best.pt和last.pt的检测结果(虽然什么都没识别出来)</p>

<h2>detect.py轻量化</h2>
<p>在light_detect.py中</p>
