终端切换到当前文件夹下输入命令即可运行

（1）使用提示
> python main.py -h

（2）测试用例
> python main.py --inference --netpath checkpoints\PINNcase3_60s_20000e\best_model.pth.tar 
（--inference仅推理不做训练；--netpath给出trained model文件路径）

> python main.py --inference --netpath checkpoints\PINN_10s_10000e\best_model.pth.tar --expt 10
（--expt设置10s预测范围，默认为60s）

> python main.py --inference --netpath checkpoints\PINNnoFBMcase1_60s_20000e\best_model.pth.tar --no_FBM
 (--no_FBM声明此模型不含有FBM)

> python main.py --inference --netpath checkpoints\singlePINNcase3_60s_20000e\best_model.pth.tar --no_PSF --eps
 (--no_PSF声明此模型不含有PSF; --eps将结果保存为eps，默认png)

（3）训练用例
> python main.py --epochs_Adam 20000 --expt 60 --seed 211 --tag case1
（--epochs_Adam设置训练次数；--expt设置时间取值范围；--seed设置全局随机种子；--tag文件夹命名标记）
（将生成PINNcase1_60s_20000e文件夹，若已经存在，则覆盖更新）

> python main.py --epochs_Adam 20000 --expt 60 --seed 211 --tag noBBMcase1 --no_BBM
（--no_BBM声明此模型不含有BBM）

> python main.py --epochs_Adam 20000 --expt 60 --seed 211 --tag case1 --debug
（--debug开启debug模式，记录梯度情况至grad文件夹）

...



