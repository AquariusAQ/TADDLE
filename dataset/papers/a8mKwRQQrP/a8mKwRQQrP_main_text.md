# ONLINE POLICY SELECTION FOR INVENTORY PROBLEMS

Anonymous authors

Paper under double-blind review

## ABSTRACT

We tackle online inventory problems where at each time period the manager makes a replenishment decision based on partial historical information in order to meet demands and minimize costs. To solve such problems, we build upon recent works in online learning and control, use insights from inventory theory and propose a new algorithm called GAPSI. This algorithm follows a new feature-enhanced base-stock policy and deals with the troublesome question of non-differentiability which occurs in inventory problems. Our method is illustrated in the context of a complex and novel inventory system involving multiple products, lost sales, perishability, warehouse-capacity constraints and lead times. Extensive numerical simulations are conducted to demonstrate the good performances of our algorithm on real-world data.

## 1 INTRODUCTION

Inventory control is a standard problem in operations research and operations management where a manager needs to make replenishment decisions in order to minimize costs and meet demands (Nahmias, 2011; Roldán et al., 2017). Classical inventory theories focus on the optimization side of the problem, that is, determining what is the optimal replenishment strategy assuming all the parameters of the inventory model are known. For instance, the manager knows future demands or the distribution they are drawn from. However, in real-world scenarios an inventory manager rather faces optimization and learning problems presented in a sequential fashion, that is, at each time period the manager makes its replenishment decisions based on past observations. For these reasons, machine learning frameworks such as online learning or reinforcement learning seem particularly well-suited to solve these realistic inventory problems. For instance, there is now an important body of literature that builds upon techniques from online convex optimization like online gradient descent (Zinkevich, 2003) to solve inventory problems (Hihat et al., 2024). However, they are extremely specialized, designed for specific inventory models involving specific dynamics, cost structures and demand processes (see, for example, Huh & Rusmevichientong, 2009; Shi et al., 2016; Zhang et al., 2018). On the other hand, other generic approaches for control problems such as Model Predictive Control (MPC) (Mattingley et al., 2011) suffer from a prohibitive computational cost and a lack of scalability.

Contributions. In this paper, we first show how realistic, general inventory problems fit within the recent Online Policy Selection (OPS) framework of Lin et al. (2024) which is at the crossroads of control and online learning. We detail how various constraints very common in the industry, such as perishability, lead times, or warehouse capacity constraints, can be mathematically modeled in this setting. We obtain a general online inventory problem, that can be simplified and applied to several more classical problems such as usual perishable inventory systems (Nahmias, 2011).

We then present a new algorithm for solving this online optimization problem, called GAPSI. This algorithm is adapted from GAPS (Lin et al., 2024) to take into account specific aspects of inventory problems, in particular the fact that the functions we are dealing with are not differentiable. We find that this non-differentiability problem cannot be ignored, since it leads to undesirable behaviors, and we show how carefully selected derivatives for the policies and Adagrad-style learning rates solve it. We also propose a new policy which draws on classical base-stock policies (Snyder & Shen, 2019, Section 4.3.1) while dealing with uncertainty in future demand. The idea is to learn a target

level which writes as a linear function of either past demand features or forecasts. In this way, we obtain a general online method for solving real-world inventory problems that can take into account many different constraints and domain-specific knowledge about demands. For example, if we know that demand has a weekly seasonality, then this knowledge can be incorporated into the algorithm by using indicators of the day of the week as features.

Finally, we provide extensive experiments which demonstrate the good performance of GAPSI compared to classical approaches. We observe that GAPSI performs particularly well when demands are not stationary, have some changes of regime, and when the features are well-chosen. We emphasise that in this work we do not only propose an efficient new algorithm, but also want to show that online learning is a promising approach for realistic inventory problems, and therefore draw the attention of this community to these problems.

Related works. Many online learning frameworks have been applied or adapted to inventory problems. For instance, in the absence of inventory dynamics, Levina et al. (2010) see their problem as prediction with expert advice and Lugosi et al. (2024) use partial monitoring. In the presence of dynamics, the situation is much more complex, and this is what interests us. The research most closely related to this paper typically considers Online Convex Optimization (OCO) (Zinkevich, 2003) as their main learning framework. This is the case of Huh & Rusmevichientong (2009), Shi et al. (2016), Zhang et al. (2018), Zhang et al. (2020), Guo et al. (2024). The common approach in this literature is to adapt OCO techniques, such as Online Gradient Descent (OGD), in various ways to handle inventory dynamics. However, this is mostly done on a case-by-case basis: specialized algorithms are designed for specific dynamics. So the manager can only implement these solutions if they are facing a very similar problem.

The reason why most inventory problems are not instances of classical online learning problems is that they lack a notion central in control problems: that of a dynamical system influenced by the manager's decisions and impacting the losses. Incorporating this notion in online learning leads us to the literature of online control. The latter has mostly focused on linear dynamics, as in Agarwal et al. (2019), but most inventory dynamics are not linear. Among the recent developments in online control, the OPS framework of Lin et al. (2024), presented in Subsection 2.1, provides much more flexibility by allowing for general dynamics. Nevertheless, some challenges remain when considering inventory problems through OPS among which the non-differentiability of the losses, policies and dynamics.

On the other hand, there exist general-purpose control techniques which are not specific to inventory problems, in particular, Model Predictive Control (MPC) (Mattingley et al., 2011), which became popular in the 1980s. The main idea is to solve an optimization problem at each time step based on a predictive model up to a receding planning horizon. These approaches are not the main subject of this paper, but we consider MPC as a competitor in the experimental section.

Overview. We start in Section 2 by introducing the OPS framework and show through various examples that it is well-adapted to model realistic inventory problems. Then, we present in Section 3 our algorithm, GAPSI, that is based on GAPS (Lin et al., 2024) while taking into account aspects specific to inventory management, via the use of a feature-enhanced base-stock policy, AdaGrad learning rates and carefully chosen partial derivatives. We conclude in Section 4 with an extensive numerical simulation study. We refer to the appendix for precise mathematical details and additional experiments.

Notations. Let us denote by  $ R_{+} = [0, +\infty) $  the set of non-negative real numbers,  $ N = \{1, 2, \ldots\} $  the set of positive integers,  $ N_{0} = \{0, 1, \ldots\} $  the set of non-negative integers and  $ [n] = \{1, \ldots, n\} $ .

## 2 PROBLEM STATEMENT

### 2.1 ONLINE POLICY SELECTION

The Online Policy Selection (OPS) framework of Lin et al. (2024) is a discrete-time control problem where the decision-maker learns the parameters of a parameterized policy in an online fashion. Formally, let X be the state space, U the control space and  $ \Theta $  the parameter space. At each time

period  $ t \in N $ , the decision-maker starts by observing the state of the system  $ x_{t} \in X $ , then, they choose a parameter  $ \theta_{t} \in \mathbb{O} $  which is used to determine the control through a time-varying policy:  $ u_{t} = \pi_{t}(x_{t}, \theta_{t}) \in \mathbb{U} $ . A loss  $ \ell_{t}(x_{t}, u_{t}) \in \mathbb{R} $  is incurred, and finally the system transitions to the next state:  $ x_{t+1} = f_{t}(x_{t}, u_{t}) \in \mathbb{X} $ . Given a horizon  $ T \in N $ , the goal is to minimize the cumulative loss  $ \sum_{t=1}^{T} \ell_{t}(x_{t}, u_{t}) $  by selecting  $ \theta_{t} $  sequentially. We assume that the dynamics  $ (f_{t})_{t \in \mathbb{N}} $ , losses  $ (\ell_{t})_{t \in \mathbb{N}} $ , and policies  $ (\pi_{t})_{t \in \mathbb{N}} $  are oblivious, meaning they are fixed before the interaction starts.

Note that the well-studied Online Convex Optimization (OCO) framework (Zinkevich, 2003) can be seen as a special case of OPS, obtained by removing the dynamics of the system, simplifying the policies, and introducing convexity assumptions. Formally, to recover OCO, one can set  $ X = \{0\} $  and  $ \pi_{t}(x_{t}, \theta_{t}) = \theta_{t} $ , assume that  $ \Theta $  is a closed convex subset of an Euclidean space and that  $ u_{t} \mapsto \ell_{t}(x_{t}, u_{t}) $  is convex for every  $ t \in N $ .

### 2.2 INVENTORY PROBLEMS

To model an inventory, we must determine its dynamics and the cost structure to be minimized. The vector  $ x_{t} $  must fully describe the state of the inventory at time t. It encodes the quantities of products, including both on-hand units and eventual on-order units. The control  $ u_{t} $  is used here to represent the ordered quantities at time t. The transitions  $ f_{t} $  will determine how the inventory states evolve over time. Loss functions  $ \ell_{t} $  determine the cost structure. We describe below how these quantities can be defined, starting with a simple model and making it more complex as we go along.

Lost sales. First, assume that we have one product and no lead time, meaning that a product is instantaneously received when ordered and no perishability. Then,  $ x_{t} \in R_{+} $  is simply the quantity of this product available in the inventory. We assume that unmet demand is lost, which is modeled by the transition  $ f_{t}(x_{t}, u_{t}) = [x_{t} + u_{t} - d_{t}]^{+} $ , where  $ d_{t} \in R_{+} $  is the demand at time period t. Online lost sales inventory problems have been investigated by Huh & Rusmevichientong (2009) for instance.

Fixed lifetime perishability. Now, if this product has a fixed usable lifetime of  $ m \in N $  periods, then a unit received on period t can be sold from periods t to  $ t + m - 1 $  and, if this does not happen, it expires and leaves the inventory at the end of period  $ t + m - 1 $  as an outdating unit. To model such a system we need to keep track of the entire age distribution of the on-hand inventory through a state vector  $ x_{t} $  of dimension n = m - 1 (Nahmias, 2011, Section 1.3). For  $ i \in [m - 1] $ , the  $ i^{th} $  coordinate  $ x_{t,i} $  represent the number of units that will expire at the end of day  $ t + i - 1 $  if not sold before. We again assume that unmet demand is lost and we choose a classical transition  $ f_{t} $  which sells first the oldest products (Nahmias, 2011, Chapter 2). This can be written component-wise as

 $$ f_{t,i}(x_{t},u_{t})=f_{t,i}(z_{t})=\left[z_{t,i+1}-\left[d_{t}-\sum_{j=1}^{i}z_{t,i}\right]^{+}\right]^{+}, $$ 

where  $  z_{t} = (x_{t,1}, \ldots, x_{t,n}, u_{t}) \in \mathbb{R}^{n+1}  $  will be thereafter called the state-control couple. Zhang et al. (2018) proposed an online control algorithm for such systems.

Order lead times. An order lead time, also known as order delay, is the difference between the reception time period and the order time period. To adapt our model to take into account order lead times on top of tracking on-hand products, we need to track on-order products in the state vector by increasing its dimension. For instance, to include a lead time  $ L \in N_{0} $  in lost sales perishable inventory systems, we can set  $ n = m + L - 1 $ . Then, the first m - 1 coordinates of the state  $ x_{t} $  evolve following (1) and the next L coordinates correspond to on-order units. To our knowledge, there exist online algorithms handling lead times (Zhang et al., 2020; Agrawal & Jia, 2022), but they consider only non-perishable products.

Multi-product system. Assume now that there are  $ K \in N $  product types indexed by  $ k \in [K] $ . Each product is perishable with lifetime  $ m_k \in N $  and has a lead time  $ L_k \in N_0 $ . We add an index k to all quantities which are product-dependent. For example, we now have K transition functions  $ \left(f_{t,k}(x_t, u_t)\right)_{k \in [K]} $ , K demands per time step  $ (d_{t,k})_{k \in [K]} $ , and the state-control couples are denoted by  $ z_t = (z_{t,k})_{k \in [K]} $  where each  $ z_{t,k} = (x_{t,k,1}, \ldots, x_{t,k,n_k}, u_{t,k}) $ . Multi-product systems

become interesting whenever losses or transitions cannot be separated per product. This happens in the presence of joint constraints like the warehouse-capacity constraints introduced next.

Warehouse-capacity constraints. Until now, we have assumed that one can store an arbitrarily large quantity of products in the inventory, but in real-world problems we may need to ensure that the warehouse's capacity is not overflowed. For instance, Shi et al. (2016) consider this kind of constraints in an online control setting, but with no lead times nor perishability. Modeling warehouse-capacity constraints can be nontrivial, in particular in presence of lead times, since the manager does not know the future demands in advance. We propose a new model imposing restrictions at reception time to prevent overflow, which we summarize next and whose details can be found in Appendix A. Given a state-control vector  $ z_{t} $ , we check whether a warehouse-capacity constraint  $ z_{t} \in V_{t} $  is satisfied or not. If the constraint is not satisfied, some operator must be applied to  $ z_{t} $  to discard some units. We propose to remove products that just arrived in the ascending order (starting from product  $ k = 1, 2, \ldots, K $ ), which yields a new state-control vector  $ \tilde{z}_{t,k} $ . This allows us to define new transitions  $ f_{t} $ , by taking any of the previously seen transitions and evaluating it at  $ \tilde{z}_{t} $  instead of  $ z_{t} $ .

Losses. The goal in inventory control can be seen as minimizing a loss  $ \ell_{t} $  writing as a sum of terms capturing different trade-offs such as over-ordering versus under-ordering, or meeting the demand while maintaining low inventory management costs. For instance, the penalty cost is a term proportional to unmet demand, which writes  $ [d_{t,k}-\sum_{i=1}^{m_{k}}\tilde{z}_{t,k,i}]^{+} $ . Similarly, the holding cost is a term proportional to on-hand units just after meeting demand,  $ [\sum_{i=1}^{m_{k}}\tilde{z}_{t,k,i}-d_{t,k}]^{+} $ . These are the most classical costs for losses in stochastic inventory problems, see, e.g., the newsvendor problem in Arrow et al. (1951) or Snyder & Shen (2019, Subsection 3.1.3). We could also incorporate usual terms such as the purchase cost (proportional to ordered units) or the outdated cost (is proportional to outdated units). We also propose a new overflow cost, proportional to discarded units due to overflow, which is specific to our model and pushes the manager to respect the warehouse-capacity constraint. In what follows, we will assume that  $ \ell_{t} $  is the sum of these five costs (see Appendix A for details).

Summary. Putting everything together, the following timeline summarizes our model in its most general form. For each time period  $ t = 1, 2, \ldots $ 

1. The manager observes the state  $ x_{t} \in \mathbb{X} $  (which is zero if t = 1).

2. The manager orders the quantities  $ u_{t} \in U $  and pays purchase costs.

3. The manager receives the units  $ (z_{t,k,m_{k}})_{k\in[K]} $ , some of which may be discarded due to the warehouse-capacity constraint, incurring overflow costs and forming a new state-control vector  $ \tilde{z}_{t} $ .

4. The demand  $ d_{t} $  is realized and met to the maximum extent possible using on-hand units  $ (\sum_{i=1}^{m_{k}}\tilde{z}_{t,k,i})_{k\in[K]} $  starting by oldest units  $ (\tilde{z}_{t,k,1})_{k\in[K]} $ . Penalty costs and holding costs are paid.

5. The next inventory state  $ x_{t+1} \in X $  is defined through the transition  $ f_{t} $ , where outdating units leave the inventory incurring an outdating cost.

## 3 THE ALGORITHM: GAPSI

Let us now introduce our new algorithm for inventory problems named Gradient-based Adaptive Policy Selection for Inventories (GAPSI). Its pseudo-code is provided in Algorithm 1. In essence, GAPSI performs an online gradient descent to update the parameters of its replenishment policy. More precisely, GAPSI follows a new feature-enhanced base-stock policy whose parameter is sequentially updated according to AdaGrad (Streeter & McMahan, 2010; Duchi et al., 2011), using an approximated gradient computed as in GAPS (Lin et al., 2024) by combining carefully chosen generalized Jacobian matrices. We detail further what each step does below.

### 3.1 DESIGNING A POLICY TAILORED TO INVENTORY PROBLEMS

The first action the decision-maker needs to perform is to pass an order based on the current state  $ x_{t} $ , according to a certain policy  $ \pi_{t} $ , which usually depends on  $ x_{t} $  and some parameter. A standard

Algorithm 1: GAPSI

1 Parameters: Learning rate factor  $ \eta > 0 $ , buffer size  $ B \in N $ , initial parameter  $ \theta_{1} \in \Theta $ .

2 Observe the initial state  $ x_{1} $ ;

3 for  $ t = 1, 2, \ldots $  do

4     Order the quantities  $ u_{t} = \pi_{t}(x_{t}, \theta_{t}) $  according to the policy  $ \pi_{t} $  (see Section 3.1);

5     Incur the loss  $ \ell_{t}(x_{t}, u_{t}) $  and observe the next state  $ x_{t+1} = f_{t}(x_{t}, u_{t}) $ ;

6     Compute the Jacobian matrices of  $ \pi_{t} $ ,  $ \ell_{t} $  and  $ f_{t} $  (see Section 3.3);

7     Compute an approximated gradient  $ g_{t} $  as in GAPS (see Section 3.2);

8     Update the parameter  $ \theta_{t+1} $  by using  $ g_{t} $  via AdaGrad (see Section 3.2);

policy for stochastic inventory problems is the base-stock policy (Snyder & Shen, 2019, Chapter 4). Such policy is parameterized by a time-varying base-stock level  $ S_{t} \in R_{+}^{K} $  which the manager tries to maintain: for each product, if the inventory position is less than the base-stock level then the difference is ordered, otherwise no order is placed. This policy is appealing because it is optimal in a certain sense for simple problems, see Snyder & Shen (2019, Section 4.5), Bu et al. (2023), and Xie et al. (2024).

For GAPSI, we introduce a feature-enhanced variant of this policy. We assume that before each order at time t, and for every product  $ k \in [K] $ , the manager has access to a vector of nonnegative features  $ w_{t,k} \in R_{+}^{p_{k}} $ . These can typically gather information about the seasonality, holidays, price discounts or demand forecasts. We then propose to follow a base-stock policy whose level  $ S_{t,k} $  is, for each product k, a linear combination of the features  $ w_{t,k,i} $  with some coefficients  $ \theta_{t,k,i} $  which we need to learn. Our choice of policy can then be formally defined as:

 $$ \pi_{t}(x_{t},\theta_{t})=\left(\pi_{t,k}(x_{t,k},\theta_{t,k})\right)_{k\in[K]}\ \mathrm{w h e r e}\ \pi_{t,k}(x_{t,k},\theta_{t,k})=\left[w_{t,k}^{\top}\theta_{t,k}-\sum_{i=1}^{m_{k}+L_{k}-1}x_{t,k,i}\right]^{+}. $$ 

Note that if we take univariate constant feature vectors (such as  $ w_{t,k} \equiv 1 $ ) we recover standard base-stock policies. We also point out that if the features are forecasts of the demand, our policy recovers an heuristic proposed by Motamedi et al. (2024, Subsection 5.1) in the context of single-product offline inventory problems. Finally, we highlight that  $ \pi_{t} $  is not differentiable, which happens to be a problem when optimizing with respect to  $ \theta $ , which will be discussed in Section 3.3.

### 3.2 LEARNING THE PARAMETERS WITH GAPS AND ADAGRAD

The main goal of GAPSI is to learn the parameters  $ \theta_{t}\in\Theta $ , which we assume to be constrained in a box  $ \Theta=\prod_{i=1}^{P}[a_{i},b_{i}] $ ,  $ P:=\sum_{k}p_{k} $ . To do so, we use an online optimization scheme which approximately minimizes a surrogate loss function  $ L_{t}:\Theta\toR $ . Precisely,  $ L_{t}(\theta) $  is the loss which we would have incurred at time t if we had followed the policy associated to  $ \theta $  for all periods so far, that is, if we had applied the controls  $ u_{s}=\pi_{s}(x_{s},\theta) $  for all  $ s=1,\ldots,t $ .

Ideally, and this is a standard idea in online control, we would like to perform an online gradient descent with respect to  $ L_{t} $ , but the cost of computing  $ \nabla L_{t}(\theta_{t}) $  is prohibitive when t grows. This is why we turn to GAPS (Lin et al., 2024), a procedure returning an approximated gradient  $ g_{t} \sim \nabla L_{t}(\theta_{t}) $  at a reasonable cost, by making two approximations. First, instead of computing  $ \nabla L_{t}(\theta_{t}) $  along the ideal trajectory of parameters  $ (\theta_{t}, \ldots, \theta_{t}) $ , it is computed along the current trajectory  $ (\theta_{1}, \ldots, \theta_{t}) $ , allowing to use efficiently past computations. Second, the historical dependency is truncated to the B most recent time steps. By doing so, all we need to do at each time step is to compute the jacobians of the functions  $ \pi_{t}, f_{t}, \ell_{t} $  and to combine them with past jacobians to calculate  $ g_{t} $ ? For more details on the implementation, we refer to Lin et al. (2024) and Appendix B.

Once we have computed the approximated gradient  $ g_{t} $ , GAPSI can update the parameter  $ \theta_{t} $  by performing one step of AdaGrad (Streeter & McMahan, 2010; Duchi et al., 2011), where the learning rates are set component-wise as in Orabona (2019, Algorithm 4.1) and can be further tuned with an extra parameter  $ \eta > 0 $ :

 $$ \theta_{t+1}=\mathrm{Proj}_{\Theta}\left(\theta_{t}-H_{t}g_{t}\right)\ \mathrm{with}\ H_{t}=\mathrm{diag}(\eta_{t,1},\cdots,\eta_{t,P})\ \mathrm{and}\ \eta_{t,i}=\eta\frac{b_{i}-a_{i}}{\sqrt{\sum_{s=1}^{t}g_{s,i}^{2}}}. $$ 

Our choice of this variant of AdaGrad is motivated by its adaptivity to the gradients, its coordinate-wise learning process and its decreasing learning rates.

### 3.3 THE TROUBLESOME COMPUTATION OF JACOBIANS FOR NONSMOOTH FUNCTIONS

As described in Section 3.2, at each iteration we need to compute the jacobians $ ^{1} $  of the functions  $ \pi_{t}, f_{t} $ , and  $ \ell_{t} $ . A striking feature of these functions is that none of them is differentiable, due to the presence of positive parts in their definition (see Sections 2.2 and 3.1). In standard machine learning, non-differentiability may cause theoretical difficulties (Bolte & Pauwels, 2021), but in practice it is usually not a problem: most neural network architectures include ReLU (the positive part function) but still perform perfectly well. This apparent contradiction can be ignored by observing that, in general, points of non-differentiability are never reached during training (Bertoin et al., 2021, Theorem 2).

However, this story appears to be surprisingly different for online inventory problems. First, different choices of subgradients can lead to drastically different trajectories for GAPSI, and some of them can lead to disastrous performance. Second, in some real-world scenarios most jacobians simply cannot be accessed. Therefore, the main message of this section is that one cannot blindly rely on automatic differentiation for such nonsmooth online problems. We explain below where those problems come from, and how to avoid them.

Differentiating the policy  $ \pi_{t} $ . Imagine a scenario in which the demand for a product is zero on a given interval of time, pushing the manager to reduce the corresponding stock to zero. Then arises the question of what happens when the demand becomes positive again. One would expect that the manager starts to order again the said product. But it appears that this depends heavily on how the partial derivatives of  $ \pi_{t} $  are computed. To see this, look at Figure 1 where we simulate a simple problem where the demand is 0 for 100 days, and then switches to 1 for the next 100 days. When running GAPSI with standard autodifferentiation rules for derivating  $ \pi_{t} $ , one can see that after the 100th day the base-stock level remains stationary: the manager keeps the inventory level to 0, missing numerous sales. A simple analysis (see Appendix B.2 for the details) shows that because autodiff computes the left-partial derivative of  $ \pi_{t} $ , as soon as  $ \theta_{t}=0 $  the parameter will remain this way even if the demand restarts. Therefore, we advocate for always taking the right-partial derivatives of  $ \pi_{t} $ . Our custom differentiation rule can be seen in action in Figure 1. Whenever the level reaches zero, our differentiation rule implies that a negative gradient is computed, therefore increasing the level. This leads to oscillations which decay thanks to the Adagrad learning rate (2). On the other hand, as soon as the level reaches zero, auto-differentiation leads to a zero gradient, independently of the demand which leads to this undesirable stationary behavior. We highlight that zero demand for a long time interval is not specific to the above toy problem, but can often be observed in real-world problems (see Figure 4 for example).

<div style="text-align: center;"><img src="imgs/img_in_chart_box_331_1040_887_1242.jpg" alt="Image" width="45%" /></div>


<div style="text-align: center;">Figure 1: Base-stock level w.r.t. time when using GAPSI, depending on the differentiation rule.</div>


Differentiating the loss  $ \ell_{t} $  and transition  $ f_{t} $ . The loss  $ \ell_{t} $  depends on the parameters at hand, structural parameters of the problem (such as unit costs or volume information), but also on the exogenous demand  $ d_{t} $ . It is a well-known issue in inventory problems that the demand  $ d_{t} $  is sometimes unknown even at the end of time t, preventing the computation of the loss  $ \ell_{t} $  or its derivatives, called

the censored demand framework. However, in this framework it is usually accepted that the manager has access to a partial information, which is the number of sales. Previous work such as Huh & Rusmevichientong (2009) observed in such a framework that even if the subdifferential of  $ \ell_{t} $  cannot be accessed, its left-partial derivatives can still be computed. This observation can be adapted to our model, at least when there are no warehouse-capacity constraints. The situation is actually exactly the same for the transition  $ f_{t} $ . We observe that its subdifferential cannot be accessed in general, but that its left-partial derivatives can be computed, which to our knowledge has never been discussed in prior works.

All derivations of these partial derivatives are given in Appendix B and implemented in the code. As a side benefit, we get a faster algorithm with these custom derivatives than if we had used autodifferentiation directly.

## 4 NUMERICAL EXPERIMENTS

In this section we evaluate empirically the performances of GAPSI. We refer to Appendix C for additional details and results.

Datasets. In this section, we use two real-world datasets, the M5 dataset provided by the company Walmart (Makridakis et al., 2022), and a proprietary dataset from the company Califrais, a food supply chain start-up. Both datasets include multiple sales time series over horizons of T = 1969 and T = 860 days respectively. The time series are organized hierarchically: the total demand is split into categories which are split into subcategories, which are finally split into products. These datasets are therefore very rich depending on where we place ourselves in the hierarchy, the total demand (root of the hierarchy) having much less variability than the demand of one specific product (leaves of the hierarchy). We will use different levels in this hierarchy, treating the time series as demand for a single product, even if it is actually aggregated over different products (of a category or all of them in the case of the total demand).

Metrics. To compare algorithms, our metric throughout the section is the ratio of cumulative losses between the considered algorithm and the best stationary base-stock policy  $ S_{T}^{*} $ . This policy picks the single best base-stock level independently of time, given the demand realizations over a horizon T. It is therefore an oracle in the sense that it sees future demands and cannot be implemented in practice, and makes an assumption of stationarity. This metric follows the standard approach in online learning, which consists in comparing an algorithm to a constant strategy. A ratio below one therefore means that the algorithm considered has a better performance than  $ S_{T}^{*} $ , and is equivalent to having a negative regret in online learning. In the appendix, we complete the results with two other metrics adapted to inventory problems, the lost sales and outdated percentage.

### 4.1 CYCLIC DEMANDS

We start with an experiment to illustrate the behavior of GAPSI against cyclic demands. We consider a single-product lost sales FIFO perishable inventory system without warehouse-capacity constraints  $ (K = 1, V_{t} = +\infty) $ . The demand of the product is given by the total demand of the M5 dataset described above, with a lifetime of two  $ (m = 2) $  and no lead times  $ (L = 0) $ . We consider time-invariant unit costs.

To test the performance of GAPSI, we need to specify a choice of policy, that is, a choice of features. To see how the performances of GAPSI can be improved using features, we enhance GAPSI with 15 features as follows:

 $$ w_{t}=\left(D_{T},D_{T}\mathbb{1}_{\{t\bmod7=0\}},\cdots,D_{T}\mathbb{1}_{\{t\bmod7=6\}},d_{t-7},\cdots,d_{t-1}\right)\in\mathbb{R}_{+}^{15}, $$ 

where  $ D_{T} $  is an a priori upper bound on the demand. The first component of  $ w_{t} $  is time-invariant and plays the role of an intercept, the next 7 components can be interpreted as the one-hot encoding of the day of the week, and the last 7 components consist of past demands. We compare the performance of GAPSI with these features and GAPSI without any feature (in which case, only the intercept  $ D_{T} $  is kept in (3)). We also include in the comparison the best cyclic base-stock policy (each day of the week has its own base-stock level).

<div style="text-align: center;"><img src="imgs/img_in_chart_box_224_169_673_464.jpg" alt="Image" width="36%" /></div>


<div style="text-align: center;">Figure 2: Demand (black curve) and base-stock levels (colored curves) of variants of GAPSI and baselines on a time window of 100 steps. In the legend, the ratio of losses is given between parenthesis for each algorithm.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_705_170_998_464.jpg" alt="Image" width="23%" /></div>


<div style="text-align: center;">Figure 3: Ratio of losses against level of variability (increasing from left to right) for 10 products selected from the M5 dataset (selected as quantiles of all products ranked by their normalized standard deviation)</div>


The results are given in Figure 2. GAPSI without features outperforms the best stationary base-stock policy (it has a ratio of losses below 1). When considering the feature vector (3) we observe that these performances are further improved. The best cyclic base-stock policy positions itself between GAPSI without features and GAPSI with features. Figure 2 also shows the different behaviors: compared to the stationary base-stock policy (blue curve) GAPSI without features (orange curve) is able to adapt its level but have slow variations and cannot learn seasonal patterns unless features are provided (red curve).

### 4.2 COMPARISON STUDY

We now conduct an extensive comparison of GAPSI on several demand dynamics and against several competitors, in particular against MPC. MPC (Mattingley et al., 2011), also known as Receding Horizon Control, is a classical approach in operational research that is based on solving at each time period an optimization problem that aims at minimizing predicted future losses up to a receding planning horizon. Formally, at each time period t, using past information, we build a predictive model  $ \hat{x}_{\tau+1|t} = \hat{f}_{\tau|t}(\hat{x}_{\tau|t}, \hat{u}_{\tau|t}) $  for  $ \tau = t, \ldots, t + H - 2 $  that is initialized with  $ \hat{x}_{t|t} = x_t $  and minimize predicted losses  $ \sum_{\tau=t}^{t+H-1} \hat{\ell}_{\tau|t}(\hat{x}_{\tau|t}, \hat{u}_{\tau|t}) $  under this predictive model. This provides a sequence of planned controls  $ \hat{u}_{t|t}, \ldots, \hat{u}_{t+H-1|t} $ , from which we execute the control  $ u_t = \hat{u}_{t|t} $ .

MPC therefore requires to have access to forecasted demands. To obtain a fair comparison, we thus take as features for GAPSI the same forecasts instead of the feature vector (3). We take as forecasts the demand of the previous week:  $ \hat{d}_{t}=d_{t-7} $ . To obtain an estimate of the stability of the different algorithms with respect to noise in the forecasts, we perturb them with independent and identically distributed Gaussian noise, which yields N=10 different forecasts.

We use the same inventory system as in the previous section and consider four different demand curves: three levels of the M5 dataset (Total, Category and Product), and the Total level of Califrais. Then, we compare the following algorithms: the MPC approach with a planning horizon of H = 7 days and planned demands equal to the forecasted demands  $ (\hat{d}_{t}, \ldots, \hat{d}_{t+6}) $ , the best stationary base-stock policy  $ S_{T}^{*} $ , the non-stationary base-stock policies with levels  $ \hat{d}_{t} $ , GAPSI without features, and GAPSI with features  $ w_{t} = (D_{T}, \hat{d}_{t}) $ .

The results are given in Table 1. They show that both the base-stock policy with levels  $ \hat{d}_{t} $  and the MPC approach have similar performances. Both are significantly worse than the baseline  $ S_{T}^{*} $  and have the highest variance. On the other hand, GAPSI is able to outperform all the other algorithms while having a small variance. This experiment also shows that GAPSI can take advantage of

<div style="text-align: center;">Table 1: Performances and robustness in terms of ratio of losses. Standard deviations are taken over 10 repetitions where we inject noise into the forecasts (GAPSI without forecasts therefore does not have standard deviations)</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td colspan="3">M5</td><td style='text-align: center;'>Califrais</td></tr><tr><td style='text-align: center;'>Total</td><td style='text-align: center;'>Category</td><td style='text-align: center;'>Product</td><td style='text-align: center;'>Total</td></tr><tr><td style='text-align: center;'>Base-stock levels  $ \hat{d}_{t} $</td><td style='text-align: center;'>1.196 \pm 0.008</td><td style='text-align: center;'>1.219 \pm 0.009</td><td style='text-align: center;'>1.028 \pm 0.011</td><td style='text-align: center;'>1.579 \pm 0.022</td></tr><tr><td style='text-align: center;'>MPC</td><td style='text-align: center;'>1.199 \pm 0.008</td><td style='text-align: center;'>1.219 \pm 0.009</td><td style='text-align: center;'>1.028 \pm 0.011</td><td style='text-align: center;'>1.579 \pm 0.021</td></tr><tr><td style='text-align: center;'>GAPSI without forecasts</td><td style='text-align: center;'>0.952</td><td style='text-align: center;'>0.944</td><td style='text-align: center;'>0.735</td><td style='text-align: center;'>0.902</td></tr><tr><td style='text-align: center;'>GAPSI with forecasts</td><td style='text-align: center;'>0.910 \pm 0.002</td><td style='text-align: center;'>0.904 \pm 0.002</td><td style='text-align: center;'>0.704 \pm 0.005</td><td style='text-align: center;'>0.912 \pm 0.003</td></tr></table>

forecasts better than the other approaches tested which tend to “overfit” by ordering just enough to meet the forecasted demand. Moreover, MPC is slower to run (around 100 times longer), while base-stock level approaches have running times of the same order of magnitude as GAPSI. Complete running times are given in Appendix C

<div style="text-align: center;"><img src="imgs/img_in_chart_box_223_595_595_839.jpg" alt="Image" width="30%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_621_590_1000_844.jpg" alt="Image" width="30%" /></div>


<div style="text-align: center;">Figure 4: Base-stock levels of GAPSI and  $ S_{T}^{*} $  against the demand of product HOUSEHOLD_1_022 whose normalized standard deviation is high (1.64) and corresponds to the ninth decile of all products. The last 969 time steps are shown.</div>


<div style="text-align: center;">Figure 5: Colored curves show the evolution of demands (solid curves) and GAPSI's base-stock levels (colored dashed curves) for each one of the 3 products (blue, yellow, green). Grey curves show the sum of GAPSI's base-stock levels (dashed curve) and the warehouse volume V (solid line).</div>


### 4.3 IMPACT OF THE VARIANCE

The larger the variability in the demands, the more difficult inventory optimisation becomes. Here we design an experiment to study the robustness of GAPSI to such variability. The parameters of this experiment are similar to the previous ones, the only differences being the demands, selected as follows. The 3049 products of the M5 dataset have been ranked in ascending order of variability. The variability is measured by their standard deviation over the horizon, normalized so that the magnitude of the demands does not affect this measure. We then select one product for each decile of the ranked products. The formula and examples of demands corresponding to different levels of variability can be found in Appendix C.

The results are given in Figures 3 and 4. In Figure 3, the ratio of losses is plotted against the level of variability, increasing from left to right. We notice that in the first six examples, which have the lowest normalized standard deviation, GAPSI incurs a cost that is close to the best stationary base-stock policy  $ S_{T}^{*} $ , with a ratio of losses ranging from 0.944 to 1.013. Then, in the last 4 examples, which have a higher normalized variance, GAPSI drastically outperforms  $ S_{T}^{*} $  with a competitive ratio ranging from 0.384 to 0.493. This phenomenon is due to the nature of these last demands which feature many consecutive periods of zero demand, a situation in which GAPSI can adjust its base-stock level by temporarily reducing it, whereas the stationary policy  $ S_{T}^{*} $  cannot. Figure

4 is an example of such demand, and we can see how GAPSI adapts to a period of zero demand (between t = 1000 and t = 1200), and also how later, in a period of demand with a positive trend, it slowly increases its base-stock level. This illustrates how GAPSI is particularly well-adapted to non-stationary demands.

### 4.4 MULTIPLE PRODUCTS AND WAREHOUSE-CAPACITY CONSTRAINTS

We conclude with an experiment with multiple products and warehouse-capacity constraints. We consider a multi-product inventory system with K = 3 products with demands taken from the M5 dataset at the category level. The parameters of the experiment are similar to the previous experiment for each product, in particular we have  $ m_{k} = 3 $  and  $ L_{k} = 0 $ , and GAPSI is run without features. The only difference is now the presence of a finite time-invariant warehouse volume  $ V_{t} = V < +\infty $ , time-invariant unit volumes  $ v_{t,k} = 1 $  and time-invariant overflow costs.

The results are given in Figure 5 where we see that GAPSI successfully satisfies the volume constraints and even saturates it which is the desired behavior since it minimizes lost sales. Indeed, the gray dashed curve is overall below the gray solid line, that is,  $ \sum_{k=1}^{K} v_{t,k} S_{t,k} = \sum_{k=1}^{K} S_{t,k} \lesssim V $ . Notice that this is happening even though demand is overall increasing.

## 5 CONCLUSION

In this paper, we used techniques from online learning and insights from inventory control theory to address realistic inventory problems. We showed that the recent framework of Online Policy Selection (OPS) (Lin et al., 2024), at the crossroads of online learning and control, is well-adapted to model complex inventory problems involving, for instance, multiple products, perishability, order lead times and warehouse-capacity constraints. To address such problems, we designed GAPSI, a new online algorithm, and showed its efficiency through extensive numerical simulations.

However, the question of theoretical guarantees remains open. The standard assumptions of OPS are not satisfied for the inventory problems we consider. Indeed, policies, losses and transitions are not differentiable so that both classical chain rules and smoothness do not hold. Furthermore, the contraction property used in Lin et al. (2024) does not seem to hold. Proving regret bounds without these assumptions is very challenging.

The work of Bolte & Pauwels (2021) could be a promising way of handling the non-differentiability issue. They develop a flexible calculus theory for a new notion of generalized derivatives, allowing to consider chain rules for functions that are not differentiable in the classical sense.