# <center>IL by RL论文阅读</center>
### <center>Author: LinusWangg</center>
### <center>[论文地址](https://arxiv.org/abs/2108.04763)</center>

> ## **写在前面**
>> 读者目前仍然专业知识有限，无法做到完全正确。
> ## **相关预备知识介绍**
>> ### **模仿学习相关**
>> 目前的模仿学习主要分为两个方向，一个方向是行为克隆算法，行为克隆算法通过监督学习的方式从专家数据中学习智能体策略，但是在专家数据受限，专家数据不全面的情况下存在有复合误差等训练误差问题，不具有很强的鲁棒性。另一个方向是逆强化学习方向，逆强化学习使用奖赏函数与强化学习智能体进行对抗，在受限的专家数据上相较于行为克隆有较好的健壮性。  
>> 模仿学习的最终目标：使得专家数据的数据分布与智能体的数据分布相近  
>> 行为克隆算法中著名的几种算法：Behavior Cloning， DAgger  
>> 逆强化学习方向中著名的几种算法：学徒学习， GAIL  
>> 
>> ### **数据分布近似计算相关**
>> 一般用于衡量不同数据分布之间的主要计算方式有KL散度、TV散度。  
>> KL散度的计算公式为：  
>> $$D_{KL}(p||q)=\sum_{i=0}^Np(x)log\frac{p(x)}{q(x)}=-\sum_{x}p(x)\frac{q(x)}{p(x)}$$
>> TV散度的使用相较于KL散度较少，TV散度叫做总变差，主要用于理论分析问题，其计算公式为：
>> $$||\mu-\nu||_{TV}=\sup_{A\subset E}|\mu(A)-\nu(A)|$$
>> 可以证明TV距离可以转化为L1距离，证明如下：  
>> 设定$\ B=\{x|\mu(x)>\nu(x)\}\ $以及$\ B^c=\{x|\mu(x)\leq\nu(x)\}$  
>> 由于A是E的子集，其中不一定包含有B的相关元素，并且随着包含B的元素变多会呈现成$\mu(x)>\nu(x)$的趋势，因此$\mu(B)-\nu(B)$相当于$\mu(A)-\nu(A)$的上确界。  
>> $$\mu(A)-\nu(A)\leq\mu(A\cap B)-\nu(A\cap B)\leq\mu(B)-\nu(B)$$
>> $$\mu(B)+\mu(B^c)=\nu(B)+\nu(B^c)=1$$
>> $$\mu(B)-\nu(B)=(1-\mu(B^c))+(1-\nu(B^c))=\nu(B^c)-\mu(B^c)$$
>> $$\therefore|\mu(B)-\nu(B)|=|\nu(B^c)-\mu(B^c)|=\frac{1}{2}\sum_{x\in E}|\mu(x)-\nu(x)|$$
>> $$\therefore||\mu-\nu||_{TV}=\frac{1}{2}\sum_{x\in E}|\mu(x)-\nu(x)|$$
>>
> ## **文章的主要贡献与方法**
>> ### **当前算法存在的问题**
>> 目前的模仿学习算法大多都比较繁琐并且难以调度与利用，并且训练一个对抗的模仿学习架构会使得参数过多导致的性能不稳定性。
>> ### **文章的贡献**
>> 本文其实主要内容是基于Wang等人在论文Random expert distillation：Imitation learning via expert policy support estimation中，上述这篇文章提出了支撑集的概念，通俗的讲，如果定义域集合$X$的某一个子集$x$可以让一个函数$f$在该子集中呈现出$f>0$，那么该子集$x$就是所谓的支撑集。放之于模仿学习的算法框架中，就可以认为$f$是reward的奖赏函数，$X$为整个状态动作$(s,a)$的数据空间，支撑集$x$就是专家的示范数据，然后通过RND以及非期望状态的奖赏塑造，将模仿学习问题缩减为强化学习问题。  
>> 作者的本篇论文在理论上证明了这一操作的合理性，并且给出了相应的样本复杂度和边界要求。同时在实验上证明了在作者所用于裁剪模仿学习的奖赏设计与原有的奖赏设计上，智能体的表现性能是成正比的，即两种奖赏智能体最终的期望回报成正比。
>> ### **文章的理论证明**
>> 1. **符号定义**  
>> $\rho^{\pi}(s,a)$代表当前$(s,a)$状态-动作对在策略$\pi$下的数据分布，$\rho^{\pi}(s)$代表状态在策略$\pi$下的数据分布，这两者之间可以通过$\rho^{\pi}(s,a)=\rho^{\pi}(s)\pi(a|s)$进行相互转化。  
>> 定义智能体的期望回报值为$V^\pi=lim_{N\longrightarrow\infty}\mathbb{E}_{\pi}[J_{N}/N|s_{1}]=\mathbb{E}_{\rho^{\pi}}[R(s,a)]$，其中$J_{N}=\sum_{t=1}^{N}R(s_{t},a_{t})$。
>> 可以简单的通过大数定律来定义$\hat\rho(s,a)=\frac{1}{N}\sum_{t=1}^{N}\mathbb{1}\{(s,a)=(s_{t},a_{t})\}$以及$\hat\rho_{S}(s)=\frac{1}{N}\sum_{t=1}^{N}\mathbb{1}\{s=s_{t}\}$。
>> 定义TV距离算子中X代表MDP过程中出现的状态集合$S$或者是出现过的状态动作集合$S\times A$。  
>> 2. **理论证明1**  
>> 这一部分的证明主要目的是为了证明对于任意的奖赏函数，下述公式中的$v$，若两者数据分布相近，则所得的期望回报相似。  
>> **引理1.** 如果对于任意的奖赏设置$v\in[0,1]^X$，都有$\mathbb{E}_{\rho_{1}}[v]-\mathbb{E}_{\rho_{2}}[v]\leq\epsilon$，那么就有$||\rho_{1}-\rho_{2}||_{TV}\leq\epsilon$。  
>> **引理2.** 如果$||\rho_{1}-\rho_{2}||_{TV}\leq\epsilon$，那么对于任意的奖赏设置$v\in[0,1]^X$，都有$\mathbb{E}_{\rho_{1}}[v]-\mathbb{E}_{\rho_{2}}[v]\leq 2 \epsilon$。  
>> **证明1&2.** 这一部分可以通过上面TV距离与L1范式的关系进行证明。  
>> **引理3.** 我们设定由不可简化的非周期性专家策略生成的马尔可夫链为$P_{E}$，定义最小的$t$值大小为$\tau_{mix}$，从而$max_{a^{'}}||\mathbf{1}(s^{'})^{T}P_{E}^{t}-\rho_{S}^{E}||_{TV}\leq\frac{1}{4}$被限制住。并且，对于任意的分布$p_{1}$都有$\sum_{t=0}^{\infty}||p_{1}^{T}P_{E}^{t}-\rho_{S}^{E}||_{TV}\leq2\tau_{mix}$。  
>> **证明3.** 根据Levin的modern Markov书中理论4.9我们可以得到$$max_{s^{'}}||\mathbb{1}(s^{'})^{T}P_{E}^{t}-\rho_{S}^{E}||_{TV}\leq C\alpha^{t}$$  
>> 我们定义专家分布为，由于专家策略具有确定性：  $$\tilde{\rho}_{t}^{s^{'}}=(\mathbf{1}(s^{'})^{T}P_{E}^{t})(s)\pi_{E}(a|s)=(\mathbf{1}(s^{'})^{T}P_{E}^{t})$$ $$\therefore ||\tilde{\rho}_{t}^{s^{'}}-\rho^{E}||_{TV}=||\mathbf{1}(s^{'})^{T}P_{E}^{t}-\rho_{S}^{E}||_{TV}$$ $$\because \tau_{mix}\leq-log_{\alpha}C-log_{\alpha}(4)$$ $$\therefore max_{a^{'}}||\mathbf{1}(s^{'})^{T}P_{E}^{t}-\rho_{S}^{E}||_{TV}\leq\frac{1}{4}$$  
>> 定义$d(t)=sup_{p}||p^{T}P_{E}^{t}-\rho_{S}^{E}||_{TV}$，在Levin书中已知$d(\mathbb{l}\tau_{mix})\leq 2^{-\mathbb{l}}$。 
>> $$\therefore \sum_{t=0}^{\infty}||p_{1}^{T}P_{E}^{t}-\rho_{S}^{E}|| \leq \sum_{t=0}^{\infty}d(t) \leq \sum_{t=0}^{\infty}d(\tau_{mix}\lfloor t/\tau_{mix} \rfloor) \leq \sum_{t=0}^{\infty}2^{-\lfloor t/\tau_{mix} \rfloor}$$
>> $$\therefore \sum_{t=0}^{\infty}2^{-\lfloor t/\tau_{mix} \rfloor}=\tau_{mix}\sum_{t=0}^{\infty}2^{-t}=2\tau_{mix}$$
>> 3. **假设与命题的提出**  
>> **奖赏函数设置**  $$R_{int}(s,a)=\mathbf{1}\{(s,a)\in D\}$$  
>> **基础假设**  
>> - 专家策略是确定性的（对于相同状态动作相同）。  
>> - 专家策略是基于非循环的，不可切分的马尔可夫决策过程的。  
>> - 模仿学习策略也是基于非循环的，不可切分的马尔可夫决策过程。   
>> **命题1**  
>> 考虑一个在$N$尺度代表限制的状态-动作分布$\rho^{I}$的数据集上进行训练的模仿学习者，在假设1-3的基础上，我们有至少$1-\delta$的概率在$N\geq max\{800|S|, 450log(\frac{2}{\delta})\}\tau_{mix}^{3}\eta^{-2}$的情况下达到$||\rho^{E}-\rho^{I}||_{TV}\leq \eta$以及在原本的奖赏情况下$\mathbb{E}_{\rho_{I}}[R] \geq \mathbb{E}_{\rho_{E}}[R]-\eta$  
>> 4. **理论证明2**  
>> **引理4.** 专家分布$\rho^{E}$以及N采样数据下的专家分布近似$\hat{\rho}$可以在至少$1-2exp\{-\frac{\epsilon^{2}N}{4.5\tau_{mix}}\}$的概率下被限制到$||\rho^{E}-\hat{\rho}||_{TV}\leq \epsilon+\sqrt{\frac{8|S|\tau_{mix}}{N}}$  
>> **证明4.** 定义$\hat{\rho}_{S}=\frac{1}{N}\sum_{t=1}^{N}\mathbf{1}\{s_{t}=s\}$，实例化Paulin(2015)提出的命题3.16得到$$\mathbb{E}_{\hat{\rho}_{S}}[||\rho_{S}^{E}-\hat{\rho}_{S}||_{TV}]\leq \sum_{s}min(\sqrt{\frac{4\rho_{S}^{E}(s)}{N\gamma_{ps}}}, \rho_{S}^{E}(s))$$ 其中$\gamma_{ps}代表决策链的伪谱差距$ 对右边的等式通过柯西不等式进行放缩得到$$\sum_{s}min(\sqrt{\frac{4\rho_{S}^{E}(s)}{N\gamma_{ps}}}, \rho_{S}^{E}(s)) \leq \sum_{s}\sqrt{\frac{4\rho_{S}^{E}(s)}{N\gamma_{ps}}} \leq \sqrt{\frac{4|S|}{N\gamma_{ps}}}$$ 根据Paulin(2015)提出的等式3.9 $\gamma_{ps} \geq \frac{1}{2\tau_{mix}}$ 我们得到$$\mathbb{E}_{\hat{\rho}_{S}}[||\rho_{S}^{E}-\hat{\rho}_{S}||_{TV}] \leq \sqrt{\frac{8|S|\tau_{mix}}{N}}$$ 实例化Paulin(2015)提出的命题2.18 $$P(|\|\rho_{S}^{E}-\hat{\rho}_{S}\|_{TV}-\mathbb{E}_{\hat{\rho}_{S}}[\|\rho_{S}^{E}-\hat{\rho}_{S}\|_{TV}]| \geq \epsilon) \leq 2exp\{-\frac{\epsilon^{2}N}{4.5\tau_{mix}}\}$$ 带入上述等式后我们可以得到 $$P(\|\rho_{S}^{E}-\hat{\rho}_{S}\|_{TV}| \geq \epsilon + \sqrt{\frac{8|N|\tau_{mix}}{N}}) \leq 2exp\{-\frac{\epsilon^{2}N}{4.5\tau_{mix}}\}$$ 其实原文这里的$2exp\{-\frac{\epsilon^{2}N}{4.5\tau_{mix}}\}$是偏大的，因为在2.18的实例化公式中是带有绝对值的，那么相应的在去绝对值的过程中应该分为了相应的两边区间，但是作者可能考虑到这两部分区间并非对称的于是保留了偏大值进行证明。  
>> **引理5.** 使用N的样本点生成一个专家直方图，至少有$1-2exp\{-\frac{\epsilon^{2}N}{4.5\tau_{mix}}\}$的概率我们能够得到一个能最大化intrinsic reward $R_{int}(s,a)=\mathbf{1}((s,a)\in D)$ 满足 $\mathbb{E}_{\rho^{I}}[R_{int}] \geq 1-\epsilon-\sqrt{\frac{8|S|\tau_{mix}}{N}}$ 的数据集  
>> **证明5.** 首先证明 $$\sum_{s,a}\rho^{E}(s,a)\mathbf{1}\{\hat{\rho}(s,a)=0\} \leq \epsilon + \sqrt{\frac{8|S|\tau_{mix}}{N}}$$ 我们设定 $\rho_{1}=\rho^{E},\rho_{2}=\hat{\rho},M={(s,a):\hat{\rho}(s,a)=0}$ 代入 $||\rho_{1}-\rho_{2}||_{TV}=\sup_{X\subset M}|\rho_{1}(X)-\rho_{2}(X)|$ 得证上式，用简单话来说就是 $||\rho_{1}-\rho_{2}||_{TV}$ 代表了这两个分布之间的差距，这之间的差距存在有三部分：1出现2没出现，2出现1没出现以及1和2均出现了但密度不同，上式只计算了2中没出现的状态动作集合在1中出现的概率（因为1没出现的情况概率为0自然计算结果也为0），自然要比三部分的总和要低，所以得到证明。而如果是在2出现的情况下计算1出现的情况，由于计算的不是密度之差，所以不能得到相同的结论，所以继续我们得到了上述的先证明的式子 $$\because 1=\sum_{s,a}\rho^{E}(s,a)=\sum_{s,a}\rho^{E}(s,a)\mathbf{1}{\hat{\rho}(s,a)=0}+\sum_{s,a}\rho^{E}(s,a)\mathbf{1}{\hat{\rho}(s,a)>0}$$ $$\therefore \sum_{s,a}\rho^{E}(s,a)\mathbf{1}{\hat{\rho}(s,a)>0} \geq 1-\epsilon-\sqrt{\frac{8|S|\tau_{mix}}{N}}$$ 从此得证了intrinsic reward在这一理论加持下的期望每步回报的下确界$1-\epsilon-\sqrt{\frac{8|S|\tau_{mix}}{N}}$。  
>> **引理6.** 假设一个智能体遍历了l步连续状态动作，定义专家的期望extrinsic每步回报为 $\mathbb{E}_{\rho^{E}}[R]$，定义 $\tilde{\rho}_{t}^{p_{1}}(s,a)=(p_{1}^{T}P_{E}^{t})(s)\pi_{E}(a|s)$代表时间t的状态-动作对分布（由p1状态分布开始运行），那么基于状态动作序列的智能体期望每步extrinsic奖赏满足 $$\tilde{V}_{p_{1}}^{\mathbb{l}}=\sum_{s,a}((\sum_{t=0}^{\mathbb{l}-1}\frac{1}{\mathbb{l}}\tilde{\rho}_{t}^{p_{1}}(s,a))R(s,a)) \geq \mathbb{E}_{\rho^{E}}[R]-\frac{4\tau_{mix}}{\mathbb{l}}$$ 
>> **证明6.** $$\because \mathbb{E}_{\rho^{E}}[R]-\mathbb{E}_{\tilde{\rho}_{t}^{p_{1}}}[R] \leq 2\|\tilde{\rho}_{t}^{p_{1}}-\rho^{E}\|_{TV}$$ $$\therefore \mathbb{E}_{\tilde{\rho}_{t}^{p_{1}}}[R] \geq \mathbb{E}_{\rho^{E}}[R]-2\|\tilde{\rho}_{t}^{p_{1}}-\rho^{E}\|_{TV}$$ $$\therefore \tilde{V}_{p_{1}}^{\mathbb{l}} \geq \frac{1}{\mathbb{l}}\sum_{i=0}^{\mathbb{l}-1}(\mathbb{E}_{\rho^{E}}[R]-2\|\tilde{\rho}_{t}^{p_{1}}-\rho^{E}\|_{TV})=\mathbb{E}_{\rho^{E}}[R]-\frac{2}{\mathbb{l}}\sum_{i=0}^{\mathbb{l}-1}\|\tilde{\rho}_{i}^{p_{1}}-\rho^{E}\|_{TV} \geq \mathbb{E}_{\rho^{E}}[R]-\frac{2}{\mathbb{l}}\sum_{i=0}^{\infty}\|\tilde{\rho}_{i}^{p_{1}}-\rho^{E}\|_{TV}$$ 根据引理3 $\sum_{t=0}^{\infty}\|p_{1}^{T}P_{E}^{t}-\rho_{S}^{E}\|_{TV} \leq 2\tau_{mix}$ 我们可以得到引理6的结论。  
>> **引理7.** 对于任意的限制于 $[0,1]$ 的extrinsic奖赏，能够在intrinsic奖赏上得到 $\mathbb{E}_{\rho^{I}}[R_{int}]=1-\kappa$ 的模仿学习者同时也能在extrinsic奖赏上以概率1达到 $\mathbb{E}_{\rho^{I}}[R] \geq (1-\kappa)(\mathbb{E}_{\rho^{E}}[R])-4\tau_{mix}\kappa$  
>> **证明7.** 定义 $T$ 为模仿学习者的轨迹长度，考虑这其中与专家轨迹相符合的子集轨迹，设定 $M_{\mathbb{l}}$ 为这种长度为 $\mathbb{l}$ 的轨迹序列的数量，定义 $\hat{V}^{i,\mathbb{l}}$ 为第 $i$ 段长度为 $l$ 的轨迹上的每步extrinsic奖赏。  
假设最糟糕情况下状态的奖赏并不属于这类序列，轨迹总的extrinsic奖赏回报 $J_{T}$ 至少是 $\sum_{\mathbb{l}}\mathbb{l}\sum_{i=1}^{M_{t}}\hat{V}^{i,\mathbb{l}}=\sum_{\mathbb{l}}\mathbb{l} M_{\mathbb{l}} \frac{1}{M_{\mathbb{l}}} \sum_{i=1}^{M_{t}}\hat{V}^{i,\mathbb{l}}$,由于 $J_{T}$ 代表整个轨迹的extrinsic奖赏且奖赏为正值， $\hat{V}$ 代表单个符合序列的奖赏，所以两边同除以 $T$ 可以得到 $$\frac{J_{T}}{T} \geq \sum_{\mathbb{l}}\frac{\mathbb{l}{M_{\mathbb{l}}}}{T}\frac{1}{M_{\mathbb{l}}}\sum_{i=1}{M_{\mathbb{l}}}\hat{V}^{i,\mathbb{l}}$$ 设定 $\Delta_{l}=|\frac{1}{M_{l}}\sum_{i=1}^{M_{l}}\hat{V}^{i,l}-\frac{1}{M_{l}}\sum_{i=1}^{M_{l}}\tilde{V}_{p(s_{1}^{l})}^{l}|$ 代表长度 $l$ 的序列的平均extrinsic回报以及期望值的差距。 $$\frac{J_{T}}{T} \geq \sum_{l}\frac{lM_{l}}{T}\frac{1}{M_{l}}\sum_{i=1}^{M_{l}}\tilde{V}_{p(s_{1}^{l})^{l}}-\sum{l}\frac{lM_{l}}{T}\Delta_{l}$$ 通过引理6我们可以得知 $$\frac{J_{T}}{T} \geq \sum_{l}\frac{lM_{l}}{T}\mathbb{E}_{\rho^{E}}[R]-\sum_{l}\frac{M_{l}}{T}4\tau_{mix}-\sum_{l}\frac{lM_{l}}{T}\Delta_{l}$$ 假设 $B$ 为与专家不符的时间步，那么B个时间步最多能将轨迹分为B+1段，由此可得 $\sum_{l}\frac{M_{l}}{T} \leq \frac{B+1}{T}$ 因此可得 $$\frac{J_{T}}{T} \geq \sum_{l}\frac{lM_{l}}{T}\mathbb{E}_{\rho^{E}}[R]-\frac{B+1}{T}4\tau_{mix}-\sum_{l}\frac{lM_{l}}{T}\Delta_{l}$$ 由于学习的遍历性以及 $T->\infty$ 所以左手边算式趋近于 $\mathbb{E}_{\rho^{I}}[R]$ ，右手边算式子 $\sum_{l}\frac{lM_{l}}{T}$ 趋近于 $\mathbb{E}_{\rho^{I}}[R_{int}]=1-\kappa$ ，$\frac{B+1}{T}$ 趋近于 $1-\mathbb{E}_{\rho^{I}}[R_{int}]$ 。这里作者用了无穷视野去证明了 $\Delta$ 趋近于0，暂未看懂，尚不做评价。最终得到了引理7的证明。  
>> **命题1的证明** $$\because \tau_{mix}=0 or \tau_{mix} \geq 1$$  $$\therefore N \geq max{32|S|\tau_{mix}(1+4\tau_{mix})^2\eta^{-2}, log(\frac{2}{\delta})18\tau_{mix}(1+4\tau_{mix})^2\eta^{-2}}$$ $$\because Lemma5$$ $$\therefore \epsilon = \frac{\eta}{2+8\tau_{mix}}$$ $$\therefore \delta \geq 2exp{-\frac{\epsilon^{2}N}{4.5\tau_{mix}}}$$ $$therefore \kappa(1+4\tau_{mix}) \leq \eta, \kappa = \epsilon+\sqrt(\frac{8|S|\tau_{mix}}{N})$$ $$\because \mathbb{E}_{\rho^{I}}[R] \geq \mathbb{E}_{\rho^{I}}[R](1-\kappa)-4\tau_{mix}\kappa$$  $$\therefore \mathbb{E}_{\rho^{E}}[R]-\mathbb{E}_{\rho^{I}}[R] \leq \mathbb{E}_{\rho^{E}}[R]-\mathbb{E}_{\rho^{E}}[R](1-\kappa)+4\tau_{mix}\kappa=\kappa\mathbb{E}_{\rho^{E}}[R]+4\tau_{mix}\kappa \leq \kappa+4\tau_{mix}\kappa \leq \eta$$
> ## **实验与结论**  
>> ### **实验结论概述**  
>> 与理论相符合，extrinsic reward与intrisic reward相符合。  
>> 实验方面性能提升有限，并不是很明显。  
> ## **未来计划**
>> 学习一下moder MDP以及这篇论文中提到的相关理论书籍


111
