# General_ner
## Usage
### Import

```
from general_ner.train_model.config import Config
from general_ner.train_model.ner import train_from_blank, static_test, predict_line, load_model_for_predict
```

### Config.yml

```
logging:
    root: INFO
```

### Train / Test / Predict

```
import os
import hao
from general_ner.train_model.config import Config
from general_ner.train_model.ner import train_from_blank, static_test, predict_line, load_model_for_predict


if __name__ == '__main__':
    # Load training config ...
    model_config = Config()
    if model_config.use_cuda:
        torch.cuda.set_device(model_config.gpu)

    train_from_blank(model_config)

    # train_from_saved(model_config)

    model_ = load_model_for_predict(model_config)

    static_test(model_config)

    # test_loss, test_acc, report = static_test(model_config)
    # test_acc = str(test_acc * 100.)[:5] + "%"
    # print("test_loss: {}".format(str(test_loss)[:6]))
    # print("test_acc: {}".format(test_acc))
    # print(report)

    texts = ["台北市防疫中心這段防疫期間，可以規定外籍移工假日，不要聚集北車，怕群聚感染漫延開來。(可用網路聚會，用手機開群組聊天)",
             "Jilly Lai 確診前因未知而趴趴走事後才確診,這是隱憂,而另種居家隔離雖罰,還趴趴走,這也是隱憂,所以有點恐怖啦但新北市府的防疫方式能穩定民心",
             "第32例移工外勞染疫新聞事件，造成人人都恐慌害怕!但我在捷運上還是看到很多人搭捷運不戴口罩的……非常時期實在好可怕。",
             "還好板橋車站沒有印尼移工的群聚∼真是太好了。第一次看到他們群聚在台北車站時，我還以為發生什麼大事，而且他們還直接坐在地上，同行的國外朋友也問我發生何事，當時我楞住了一下只能回說不知道。現在發生第32例的事,政府也應該好好處理非法外勞與外勞群聚的地點，替他們找個開放空間又不影響交通與觀感處。"]
    for text in texts:
        word, spend = predict_line(model_config, model_, text)
        print(word)
        print(spend)

# 创建自己的data目录
    data/bert   存放下载好的预训练模型
    data/train_dev_test/labels.txt, train.txt, dev.txt, test.txt

# 有看不懂的，直接去看general/train_model/config.py中的文件
```

## Install

```
pip install general_ner
```