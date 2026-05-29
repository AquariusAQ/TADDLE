

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

## REFERENCES

Naman Agarwal, Brian Bullins, Elad Hazan, Sham Kakade, and Karan Singh. Online control with adversarial disturbances. In International Conference on Machine Learning, pp. 111–119. PMLR, 2019.

Shipra Agrawal and Randy Jia. Learning in structured mdps with convex cost functions: Improved regret bounds for inventory management. Operations Research, 2022.

Kenneth J Arrow, Theodore Harris, and Jacob Marschak. Optimal inventory policy. Econometrica: Journal of the Econometric Society, pp. 250–272, 1951.

David Bertoin, Jérôme Bolte, Sébastien Gerchinovitz, and Edouard Pauwels. Numerical influence of relu'(0) on backpropagation. Advances in Neural Information Processing Systems, 34:468–479, 2021.

Jérôme Bolte and Edouard Pauwels. Conservative set valued fields, automatic differentiation, stochastic gradient methods and deep learning. Mathematical Programming, 188:19–51, 2021.

Jinzhi Bu, Xiting Gong, and Xiuli Chao. Asymptotic optimality of base-stock policies for perishable inventory systems. Management Science, 69(2):846–864, 2023.

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7), 2011.

Sichen Guo, Cong Shi, Chaolin Yang, and Christos Zacharias. An online mirror descent learning algorithm for multiproduct inventory systems. Available at SSRN 4806687, 2024.

Massil Hihat, Stéphane Gaïffas, Guillaume Garrigos, and Simon Bussy. Online inventory problems: beyond the iid setting with online convex optimization. Advances in Neural Information Processing Systems, 36, 2024.

Woonghee Tim Huh and Paat Rusmevichientong. A nonparametric asymptotic analysis of inventory planning with censored demand. Mathematics of Operations Research, 34(1):103–123, 2009.

Tatsiana Levina, Yuri Levin, Jeff McGill, Mikhail Nediak, and Vladimir Vovk. Weak aggregating algorithm for the distribution-free perishable inventory problem. Operations Research Letters, 38(6):516–521, 2010.

Yiheng Lin, James A Preiss, Emile Anand, Yingying Li, Yisong Yue, and Adam Wierman. Online adaptive policy selection in time-varying systems: No-regret via contractive perturbations. Advances in Neural Information Processing Systems, 36, 2024.

Gábor Lugosi, Mihalis G Markakis, and Gergely Neu. On the hardness of learning from censored and nonstationary demand. INFORMS Journal on Optimization, 6(2):63–83, 2024.

Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. The m5 competition: Background, organization, and implementation. International Journal of Forecasting, 38(4):1325–1336, 2022.

Jacob Mattingley, Yang Wang, and Stephen Boyd. Receding horizon control. IEEE Control Systems Magazine, 31(3):52–65, 2011.

Maryam Motamedi, Douglas G Down, and Na Li. Optimal ordering policy for perishable products by incorporating demand forecasts. Available at SSRN 4807963, 2024.

Steven Nahmias. Perishable inventory systems, volume 160. Springer Science & Business Media, 2011.

Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213, 2019.

Raúl F Roldán, Rosa Basagoiti, and Leandro C Coelho. A survey on the inventory-routing problem with stochastic lead times and demands. Journal of Applied Logic, 24:15–24, 2017.

Cong Shi, Weidong Chen, and Izak Duenyas. Nonparametric data-driven algorithms for multiproduct inventory systems with censored demand. Operations Research, 64(2):362–370, 2016.

Lawrence V Snyder and Zuo-Jun Max Shen. Fundamentals of supply chain theory. John Wiley & Sons, 2019.

Matthew Streeter and H Brendan McMahan. Less regret via online conditioning. arXiv preprint arXiv:1002.4862, 2010.

Yaqı Xie, Will Ma, and Linwei Xin. Vc theory for inventory policies. arXiv preprint arXiv:2404.11509, 2024.

Huanan Zhang, Xiuli Chao, and Cong Shi. Perishable inventory systems: Convexity results for base-stock policies and learning algorithms under censored demand. Operations Research, 66(5):1276–1286, 2018.

Huanan Zhang, Xiuli Chao, and Cong Shi. Closing the gap: A learning algorithm for lost-sales inventory systems with lead times. Management Science, 66(5):1962–1980, 2020.

Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th international conference on machine learning (icml-03), pp. 928–936, 2003.



# Supplementary Material for “Online policy selection for inventory problems”

## Table of Contents

A Inventory problems 13  
A.1 Complete expression of transition functions and losses .13  
A.2 Equivalent expressions for discarding .14  
B GAPSI 16  
B.1 The GAPS algorithm .16  
B.2 On the selection of derivatives .18  
B.3 Definitions and properties of one-sided derivatives .18  
B.4 A chain rule for the positive part .19  
B.5 One-sided derivatives for our model .20  
B.6 Censored demand case .23  
C Numerical experiments 25  
C.1 Additional results for Section 4.1 .26  
C.2 Additional results for Section 4.2 .27  
C.3 Additional result for Section 4.3 .27  
C.4 Classical perishable inventory systems .27  
C.5 Impact of the lifetime and the lead time .29  
C.6 Large scale experiments .30

## A INVENTORY PROBLEMS

### A.1 COMPLETE EXPRESSION OF TRANSITION FUNCTIONS AND LOSSES

Order lead times. The transition function with order lead times writes

 $$ \begin{aligned}&f_{t,i}(x_{t},u_{t})=\left[z_{t,i+1}-\left[d_{t}-\sum_{i^{\prime}=1}^{i}z_{t,i^{\prime}}\right]^{+}\right]^{+}\\&f_{t,i}(x_{t},u_{t})=z_{t,i+1}\\ \end{aligned} $$ 

 $$  for i=1,\cdots,m-1, $$ 

 $$  for i=m,\cdots,m+L-1. $$ 

Multi-product system with warehouse-capacity constraints. Let us provide two examples of models handling warehouse-capacity constraints with multiple products. The first one is adapted to a simplified setting without lead-times and with non-perishable products. The second one is the one we consider. Assume there are  $ K \in N $  product types indexed by  $ k \in [K] $ .

• Shi et al. (2016) consider a lost sales system with instantaneous replenishment  $ (L = 0) $  and non-perishable products, that is, every product follows the following transition:

 $$ f_{t}(x_{t},u_{t})=\left(f_{t,k}(x_{t},u_{t})\right)_{k\in[K]}=\left(\left[x_{t,k}+u_{t,k}-d_{t,k}\right]^{+}\right)_{k\in[K]}. $$ 

However, instead of allowing the manager to choose arbitrary order quantities  $ u_{t} \in R_{+}^{K} $ , they are restricted to choose  $ u_{t} \in R_{+}^{K} $  such that:  $ \sum_{k=1}^{K}(x_{t,k} + u_{t,k}) \leq V $ , where V > 0

is the warehouse capacity. Notice that the constraint can be expressed as  $ z_{t} \in V $ , where  $ z_{t} = (x_{t,k}, u_{t,k})_{k \in [K]} $  and,

 $$ \mathbb{V}=\Big\{z_{t}=(x_{t,k},u_{t,k})_{k\in[K]}\bigm|\sum_{k=1}^{K}(x_{t,k}+u_{t,k})\leq V\Big\}. $$ 

• Consider the multi-product setting where each product  $ k \in [K] $  is perishable with lifetime  $ m_k \in N $  and have a lead time  $ L_k \in N_0 $ . The warehouse-capacity constraint is satisfied if and only if  $ z_t \in V_t $ , where

 $$ \mathbb{V}_{t}=\left\{z_{t}\mid\sum_{k=1}^{K}\sum_{i=1}^{m_{k}}v_{t,k}z_{t,k,i}\leq V_{t}\right\}, $$ 

where for a time period t,  $ v_{t,k} $  is the unit volume of product k and  $ V_{t} $  is the total volume of the warehouse. If the constraint is not satisfied, we remove products that just arrived in the ascending order (starting from product  $ k = 1, 2, \ldots, K $ ). Formally, this corresponds to defining  $ \tilde{z}_{t,k,m_{k}} $  as follows:

 $$ \left[z_{t,k,m_{k}}-\frac{1}{v_{t,k}}\left[\left[\sum_{k^{\prime}=1}^{K}\sum_{i^{\prime}=1}^{m_{k^{\prime}}}v_{t,k^{\prime}}z_{t,k^{\prime},i^{\prime}}-V_{t}\right]^{+}-\sum_{k^{\prime}=1}^{k-1}v_{t,k^{\prime}}z_{t,k^{\prime},m_{k^{\prime}}}\right]^{+}\right]^{+}, $$ 

and  $ \tilde{z}_{t,k,i}=z_{t,k,i} $  for all  $ i\in[m_{k}+L_{k}]\setminus\{m_{k}\} $ . The reader may find Equation (4) hard to interpret, so we provide in Appendix A.2 an alternative expression which is more easily interpretable.

Finally, to obtain the next state  $ x_{t+1} $  we simply need to evaluate, for every product k, the transition on the resulting vector  $ \tilde{z}_{t,k} $  rather than  $ z_{t,k} $ . That is,

 $$ f_{t,k,i}(x_{t},u_{t})=\begin{cases}\left[\tilde{z}_{t,k,i+1}-\left[d_{t,k}-\sum_{i^{\prime}=1}^{i}\tilde{z}_{t,k,i^{\prime}}\right]^{+}\right]^{+}\mathrm{for}i=1,\cdots,m_{k}-1,\\\tilde{z}_{t,k,i+1}\mathrm{for}i=m_{k},\cdots,m_{k}+L_{k}-1.\end{cases} $$ 

Notice that setting  $ V_{t} = +\infty $  removes the warehouse-capacity constraints. Finally, we assume that  $ m_{k} + L_{k} \geq 2 $  for all products  $ k \in [K] $ , that is, we never have both  $ m_{k} = 1 $  and  $ L_{k} = 0 $ , allowing us to avoid dealing with the degenerate case where the dimension of a product's state is zero ( $ n_{k} = m_{k} + L_{k} - 1 = 0 $ ).

Loss. The complete loss function writes as

 $$ \begin{align*}\ell_{t}(x_{t},u_{t})=\sum_{k=1}^{K}\Bigg(&c_{t,k}^{\mathrm{pena}}\cdot\Big[d_{t,k}-\sum_{i=1}^{m_{k}}\tilde{z}_{t,k,i}\Big]^{+}+c_{t,k}^{\mathrm{hold}}\cdot\Big[\sum_{i=1}^{m_{k}}\tilde{z}_{t,k,i}-d_{t,k}\Big]^{+}+c_{t,k}^{\mathrm{purc}}\cdot u_{t,k}\\&+c_{t,k}^{\mathrm{outd}}\cdot[\tilde{z}_{t,k,1}-d_{t,k}]^{+}+c_{t,k}^{\mathrm{over}}\cdot(z_{t,k,m_{k}}-\tilde{z}_{t,k,m_{k}})\Bigg),\end{align*} $$ 

where  $ c_{t,k}^{pena} $ ,  $ c_{t,k}^{hold} $ ,  $ c_{t,k}^{purc} $ ,  $ c_{t,k}^{outd} $ ,  $ c_{t,k}^{over} \geq 0 $  corresponds to unit costs for product k at time period t, for each respective type of cost.

### A.2 EQUIVALENT EXPRESSIONS FOR DISCARDING

In this appendix we give an alternative expression for  $ \tilde{z}_{t} $ , the state-control vector after discarding, defined in Equation (4) and provide its interpretation afterwards.

Proposition A.1. Let  $ o_{t} $  denote the volume overflow defined as:

 $$ o_{t}=\left[\sum_{k^{\prime}=1}^{K}\sum_{i^{\prime}=1}^{m_{k^{\prime}}}v_{t,k^{\prime}}z_{t,k^{\prime},i^{\prime}}-V_{t}\right]^{+}. $$ 

Also, define recursively the volume to remove for $k = 1, \ldots, K$,

 $$ r_{t,k}=\min\bigg\{v_{t,k}z_{t,k,m_{k}}\;,\bigg[o_{t}-\sum_{k^{\prime}=1}^{k-1}r_{t,k^{\prime}}\bigg]^{+}\bigg\}. $$ 

Then, we have,

 $$ \sum_{k^{\prime}=1}^{k}r_{t,k^{\prime}}=\min\Bigg\{\sum_{k^{\prime}=1}^{k}v_{t,k^{\prime}}z_{t,k^{\prime},m_{k^{\prime}}},o_{t}\Bigg\}. $$ 

Furthermore,  $ \tilde{z}_{t,k,m_{k}} $  defined in (4), can be rewritten as:

 $$ \tilde{z}_{t,k,m_{k}}=z_{t,k,m_{k}}-\frac{r_{t,k}}{v_{t,k}}. $$ 

Proof. We start by proving Equation (9) using an induction over  $ k = 1, \ldots, K $ . First, for k = 1, we have:

 $$ \sum_{k^{\prime}=1}^{1}r_{t,k^{\prime}}=r_{t,1}\stackrel{(8)}{=}\min\left\{v_{t,1}z_{t,1,m_{1}}\;,\boldsymbol{o}_{t}\right\}=\min\Bigg\{\sum_{k^{\prime}=1}^{1}v_{t,k^{\prime}}z_{t,k^{\prime},m_{k^{\prime}}}\;,\boldsymbol{o}_{t}\Bigg\}. $$ 

Now, assume Equation (9) holds for some  $ k \in [K - 1] $ . Then,

 $$ \begin{aligned}&\sum_{k^{\prime}=1}^{k+1}r_{t,k^{\prime}}\\&=\sum_{k^{\prime}=1}^{k}r_{t,k^{\prime}}+\min\left\{v_{t,k+1}z_{t,k+1,m_{k+1}},\left[o_{t}-\sum_{k^{\prime}=1}^{k}r_{t,k^{\prime}}\right]^{+}\right\}\\&\quad(using Equation(8))\\&=\sum_{k^{\prime}=1}^{k}r_{t,k^{\prime}}+\min\left\{v_{t,k+1}z_{t,k+1,m_{k+1}},\left[\underbrace{o_{t}-\min\left\{\sum_{k^{\prime}=1}^{k}v_{t,k^{\prime}}z_{t,k^{\prime},m_{k^{\prime}}},o_{t}\right\}}_{>\geq0}\right]^{+}\right\}\\&\quad(using Equation(9))\\&=\sum_{k^{\prime}=1}^{k}r_{t,k^{\prime}}+\min\left\{v_{t,k+1}z_{t,k+1,m_{k+1}},o_{t}-\min\left\{\sum_{k^{\prime}=1}^{k}v_{t,k^{\prime}}z_{t,k^{\prime},m_{k^{\prime}}},o_{t}\right\}\right\}\\&=\min\left\{\sum_{k^{\prime}=1}^{k}r_{t,k^{\prime}}+v_{t,k+1}z_{t,k+1,m_{k+1}},\sum_{k^{\prime}=1}^{k}r_{t,k^{\prime}}+o_{t}-\min\left\{\sum_{k^{\prime}=1}^{k}v_{t,k^{\prime}}z_{t,k^{\prime},m_{k^{\prime}}},o_{t}\right\}\right\}\\&=\min\left\{\min\left\{\sum_{k^{\prime}=1}^{k}v_{t,k^{\prime}}z_{t,k^{\prime},m_{k^{\prime}}},o_{t}\right\}+v_{t,k+1}z_{t,k+1,m_{k+1}},o_{t}\right\}\\&\quad(using Equation(9))\\&=\min\left\{\sum_{k^{\prime}=1}^{k+1}v_{t,k^{\prime}}z_{t,k^{\prime},m_{k^{\prime}}},o_{t}+v_{t,k+1}z_{t,k+1,m_{k+1}},o_{t}\right\}\\&=\min\left\{\sum_{k^{\prime}=1}^{k+1}v_{t,k^{\prime}}z_{t,k^{\prime},m_{k^{\prime}}},o_{t}\right\},\end{aligned} $$ 

where the last equality comes from the fact that  $ v_{t,k+1}z_{t,k+1,m_{k+1}} \geq 0 $ . This completes the proof of Equation (9).

Now we move to the proof of Equation (10).

 $$ \begin{aligned}z_{t,k,m_{k}}-\frac{r_{t,k}}{v_{t,k}}&=z_{t,k,m_{k}}-\min\left\{z_{t,k,m_{k}}\;,\frac{1}{v_{t,k}}\bigg[o_{t}-\sum_{k^{\prime}=1}^{k-1}r_{t,k^{\prime}}\bigg]^{+}\right\}\\&\quad(using Equation (8))\\&=\left[z_{t,k,m_{k}}-\frac{1}{v_{t,k}}\bigg[o_{t}-\sum_{k^{\prime}=1}^{k-1}r_{t,k^{\prime}}\bigg]^{+}\right]^{+}\\&=\left[z_{t,k,m_{k}}-\frac{1}{v_{t,k}}\bigg[o_{t}-\min\left\{\sum_{k^{\prime}=1}^{k-1}v_{t,k^{\prime}}z_{t,k^{\prime},m_{k^{\prime}}},o_{t}\right\}\bigg]^{+}\right]^{+}\\&\quad(using Equation (9))\\&=\left[z_{t,k,m_{k}}-\frac{1}{v_{t,k}}\bigg[o_{t}-\sum_{k^{\prime}=1}^{k-1}v_{t,k^{\prime}}z_{t,k^{\prime},m_{k^{\prime}}}\bigg]^{+}\right]^{+}\\&=(4),\end{aligned} $$ 

where we used in the second and fourth equality that  $ [a-b]^{+}=a-\min\{a,b\} $ , for any  $ a,b\inR $ . ☐

In light of Proposition A.1, it becomes easier to interpret the state-control vector after discarding  $ \tilde{z}_{t} $  defined by Equation (4). Indeed, at the reception of the products, one first checks the volume overflow  $ o_{t} $  defined by Equation (7). If  $ o_{t}=0 $  there is no overflow and we have  $ \tilde{z}_{t}=z_{t} $ , otherwise we remove some products. In this case, we start discarding from the first product  $ (k=1) $ . A volume  $ v_{t,1}z_{t,1,m_{1}} $  of product 1 just arrived to the warehouse and we remove a maximum of it to the extent of the volume overflow  $ o_{t} $ , that is, we remove a volume  $ r_{t,1}=\min\{v_{t,1}z_{t,1,m_{1}},o_{t}\} $  as defined in Equation (8). If  $ r_{t,1}=o_{t} $  then we do not discard products anymore, i.e.  $ r_{t,k}=0 $  for  $ k\in[K]\setminus\{1\} $ . Otherwise,  $ r_{t,1}=v_{t,1}z_{t,1,m_{1}} $ , and we move to the second product  $ (k=2) $  from which we want to remove a volume of  $ o_{t}-r_{t,1}\geq0 $  out of the volume  $ v_{t,2}z_{t,2,m_{2}} $  that just arrived, this defines  $ r_{t,2} $  according to Equation (8), and so on and so forth. Finally, from each product k we removed a volume  $ r_{t,k} $  from the quantity  $ z_{t,k,m_{k}} $  and this completely defines  $ \tilde{z}_{t,k,m_{k}} $  through Equation (10).

## B GAPSI

In this appendix we provide details on GAPSI. Section B.1 recalls how GAPS (Lin et al., 2024) and its gradient approximation procedure works. Then, Section B.2 motivates theoretically the use of custom derivative selections. Sections B.3 and B.4 state general definitions and properties related to one-sided derivatives and functions that are composed of positive parts. Finally, Sections B.5 and B.6 provide, in our new model, the formulas for the derivatives of the functions involved, in the general case and censored demand case respectively.

### B.1 THE GAPS ALGORITHM

Main ideas. The GAPS algorithm (Lin et al., 2024) solves OPS problems using an online gradient descent approach with an approximated gradient. More precisely, define the surrogate functions  $ L_{t}: \mathbb{O} \to \mathbb{R} $  where  $ L_{t}(\theta) $  is the loss incurred at time period t if we had followed the policy associated to  $ \theta $  for all periods, that is, if we applied the controls  $ u_{s} = \pi_{s}(x_{s}, \theta) $  for all  $ s = 1, \ldots, t $ . An idealized gradient descent would be with respect to these losses  $ (L_{t})_{t \geq 1} $ , taking the form:

 $$ \theta_{t+1}=\operatorname{P r o j}_{\mathbb{O}}(\theta_{t}-\eta_{t}\nabla L_{t}(\theta_{t})) $$ 

where  $ \Theta $  is the set of parameters,  $ Proj_{\Theta} $  denotes the Euclidean projection operator on  $ \Theta $  and  $ \eta_{t} $  are learning rates.

However, the complexity of computing the gradients of  $ L_{t} $  exactly grows proportionally to t, becoming intractable when the horizon is large. In GAPS, two approximations are made to compute an approximated gradient:

1. Instead of computing the gradient  $ \nabla L_{t}(\theta_{t}) $  along the ideal trajectory  $ \theta_{t},\ldots,\theta_{t} $ , it is computed along the current trajectory  $ \theta_{1},\ldots,\theta_{t} $ .

2. The historical dependence is truncated to the B most recent time steps.

Implementation. In practice, the approximated gradient can be computed in an online fashion using chain rules as detailed in Lin et al. (2024, Appendix B, Algorithm 2).

Since  $ L_{t}(\theta_{t})=\ell_{t}(x_{t}(\theta_{t}),u_{t}(\theta_{t})) $  where  $ x_{t}(\theta) $  and  $ u_{t}(\theta)=\pi_{t}(x_{t}(\theta),\theta) $  denote the state and control if we had applied the parameter  $ \theta $  from period 1 to t, then,

 $$ \begin{align*}\frac{\partial L_{t}(\theta_{t})}{\partial\theta_{t}}=&\left(\frac{\partial\ell_{t}(x_{t}(\theta_{t}),u_{t}(\theta_{t}))}{\partial x_{t}}+\frac{\partial\ell_{t}(x_{t}(\theta_{t}),u_{t}(\theta_{t}))}{\partial u_{t}}\cdot\frac{\partial\pi_{t}(x_{t}(\theta_{t}),\theta_{t})}{\partial x_{t}}\right)\cdot\frac{\partial x_{t}(\theta_{t})}{\partial\theta_{t}}\\&+\frac{\partial\ell_{t}(x_{t}(\theta_{t}),u_{t}(\theta_{t}))}{\partial u_{t}}\cdot\frac{\partial\pi_{t}(x_{t}(\theta_{t}),\theta_{t})}{\partial\theta_{t}}.\end{align*} $$ 

Applying the first approximation, that is, replacing  $ x_{t}(\theta_{t}) $  and  $ u_{t}(\theta_{t}) $  by  $ x_{t} $  and  $ u_{t} $  respectively, we obtain a first expression for the approximated gradient:

 $$ \left(\frac{\partial\ell_{t}(x_{t},u_{t})}{\partial x_{t}}+\frac{\partial\ell_{t}(x_{t},u_{t})}{\partial u_{t}}\cdot\frac{\partial\pi_{t}(x_{t},\theta_{t})}{\partial x_{t}}\right)\cdot\frac{\partial x_{t}(\theta_{t})}{\partial\theta_{t}}+\frac{\partial\ell_{t}(x_{t},u_{t})}{\partial u_{t}}\cdot\frac{\partial\pi_{t}(x_{t},\theta_{t})}{\partial\theta_{t}}. $$ 

 $$ \begin{aligned}Now,let us deal with&\partial x_{t}(\theta_{t})/\partial\theta_{t}.Since x_{t}(\theta_{t})=f_{t-1}(x_{t-1}(\theta_{t}),u_{t-1}(\theta_{t})),we have that:\\\frac{\partial x_{t}(\theta_{t})}{\partial\theta_{t}}=&\left(\frac{\partial f_{t-1}(x_{t-1}(\theta_{t}),u_{t-1}(\theta_{t}))}{\partial x_{t-1}}+\frac{\partial f_{t-1}(x_{t-1}(\theta_{t}),u_{t-1}(\theta_{t}))}{\partial u_{t-1}}\cdot\frac{\partial\pi_{t-1}(x_{t-1}(\theta_{t}),\theta_{t})}{\partial x_{t-1}}\right)\cdot\frac{\partial x_{t-1}(\theta_{t})}{\partial\theta_{t-1}}\\&+\frac{\partial f_{t-1}(x_{t-1}(\theta_{t}),u_{t-1}(\theta_{t}))}{\partial u_{t-1}}\cdot\frac{\partial\pi_{t-1}(x_{t-1}(\theta_{t}),\theta_{t})}{\partial\theta_{t-1}}.\end{aligned} $$ 

Applying the first approximation, that is, replacing the points  $  x_{t-1}(\theta_{t})  $ ,  $  u_{t-1}(\theta_{t})  $  and  $ \theta_{t} $  by  $ x_{t-1} $ ,  $ u_{t-1} $  and  $ \theta_{t-1} $  respectively, we obtain the following approximating expression for  $  \partial x_{t}(\theta_{t})/\partial\theta_{t}  $ ,

 $$ \begin{aligned}&\left(\frac{\partial f_{t-1}(x_{t-1},u_{t-1})}{\partial x_{t-1}}+\frac{\partial f_{t-1}(x_{t-1},u_{t-1})}{\partial u_{t-1}}\cdot\frac{\partial\pi_{t-1}(x_{t-1},\theta_{t-1})}{\partial x_{t-1}}\right)\cdot\frac{\partial x_{t-1}(\theta_{t-1})}{\partial\theta_{t-1}}\\&+\frac{\partial f_{t-1}(x_{t-1},u_{t-1})}{\partial u_{t-1}}\cdot\frac{\partial\pi_{t-1}(x_{t-1},\theta_{t-1})}{\partial\theta_{t-1}}.\end{aligned} $$ 

Together with the second approximation, this expression leads to a recursive way of approximating  $ \partial x_{t}(\theta_{t})/\partial\theta_{t} $  and thus  $ \nabla L_{t}(\theta_{t}) $  which is detailed below in Algorithm 2 with our notations (see also Lin et al. (2024, Appendix B, Algorithm 2)).

Algorithm 2: Gradient approximation procedure

1 $\frac{\partial x_{t}}{\partial\theta_{t-1}} \leftarrow \frac{\partial f_{t-1}}{\partial u_{t-1}} \cdot \frac{\partial\pi_{t-1}}{\partial\theta_{t-1}}$;
2 for $b = 2, \ldots, B - 1$ do
3 $\left\lfloor \frac{\partial x_{t}}{\partial\theta_{t-b}} \leftarrow \left( \frac{\partial f_{t-1}}{\partial x_{t-1}} + \frac{\partial f_{t-1}}{\partial u_{t-1}} \cdot \frac{\partial\pi_{t-1}}{\partial x_{t-1}} \right) \cdot \frac{\partial x_{t-1}}{\partial\theta_{t-b}} \right\rfloor$;
4 return $\frac{\partial\ell_{t}}{\partial u_{t}} \cdot \frac{\partial\pi_{t}}{\partial\theta_{t}} + \left( \frac{\partial\ell_{t}}{\partial x_{t}} + \frac{\partial\ell_{t}}{\partial u_{t}} \cdot \frac{\partial\pi_{t}}{\partial x_{t}} \right) \cdot \sum_{b=1}^{B-1} \frac{\partial x_{t}}{\partial\theta_{t-b}}$;

Theoretical guarantees. Under suitable assumptions Lin et al. (2024) are able to prove theoretical guarantees for GAPS in the form of regret upper bounds.

Assuming, amongst other things, Lipschitz continuity, smoothness, a contractive perturbation property, the convexity of  $ L_{t} $ , a large enough horizon T and buffer length B and appropriately tuned learning rates, Corollary 3.4 of Lin et al. (2024) gives in particular for any  $ \theta \in \Theta $ ,

 $$ \sum_{t=1}^{T}\ell_{t}(x_{t},u_{t})-L_{t}(\theta)=O(\sqrt{T}), $$ 

where  $ O(\cdot) $  hides problem-dependent constants.

### B.2 ON THE SELECTION OF DERIVATIVES

In this appendix, we illustrate on a simple theoretical example a problematic behavior of GAPSI when the derivatives or not carefully selected.

Proposition B.1. Consider any single-product inventory problem without dynamics and assume GAPSI follows a base-stock policy. Assume further that it is implemented using auto-differentiation with  $ \mathrm{ReLU}^{\prime}(0)=0 $  or using left derivatives for the policy. Then, if the base-stock level reaches zero it remains so for all the subsequent periods.

Proof. Consider a single-product inventory problem with scalar states, controls and parameters where there are no dynamics  $ (f_{t} \equiv 0) $  and arbitrary losses  $ \ell_{t} $ . Assume GAPSI follows the base-stock policy:  $ \pi_{t}(x_{t}, \theta_{t}) = [\theta_{t} - x_{t}]^{+} $  over  $ \Theta = [0, 1] $  and that we set  $ \frac{\partial \pi_{t}}{\partial \theta_{t}}(x_{t}, \theta_{t}) \leftarrow 1_{\{\theta_{t} > x_{t}\}} $  which holds if auto-differentiation is used with  $ ReLU'(0) = 0 $  or if left derivatives are used for the policy. Since there are no dynamics, following line 4 of Algorithm 2 (or line 10 of Algorithm 2 in Appendix B of (Lin et al., 2024)), we have:

 $$ g_{t}=\frac{\partial\ell_{t}}{\partial u_{t}}(x_{t},u_{t})\cdot\frac{\partial\pi_{t}}{\partial\theta_{t}}(x_{t},\theta_{t})=\frac{\partial\ell_{t}}{\partial u_{t}}(x_{t},u_{t})\cdot\mathbb{1}_{\{\theta_{t}>x_{t}\}} $$ 

Therefore for any  $ t \in N $ ,  $ \theta_{t} = 0 $  implies  $ g_{t} = 0 $ , which in turn implies  $ \theta_{t+1} = \operatorname{Proj}_{\Theta}(\theta_{t} - H_{t}g_{t}) = 0 $  and so on and so forth.

The undesirable stationary behavior described in Proposition B.1 can also be observed in practice (see Figure 1). One may wonder whether this is happening because of a wrong application of the chain rule (which is applied here on functions that are not differentiable everywhere). In fact, this is not the case, since according to the base-stock policy  $ \pi_{t}(x_{t},\theta_{t})=0 $  for all  $ \theta_{t}\leq x_{t} $ , this behavior still happens even if we computed the exact left derivative of  $ \bar{\ell}_{t}(\theta)=\ell_{t}(x_{t},\pi_{t}(x_{t},\theta)) $  and performed an (exact) online subgradient descent with respect to these derivatives. Notice that even if  $ \ell_{t}(x_{t},\cdot) $  is convex,  $ \bar{\ell}_{t} $  is not necessarily convex which explains the difficulties encountered by a subgradient descent. Due to this undesirable behavior we advocate for the use of right derivatives instead of left derivatives for the policies.

### B.3 DEFINITIONS AND PROPERTIES OF ONE-SIDED DERIVATIVES

First, we start by some definitions regarding one-sided and standard derivatives.

Definition B.2. Let  $ f : R \to R $  and  $ x \in R $ . We denote by

 $$ \partial^{+}f(x)=\lim_{h\to0^{+}}\frac{f(x+h)-f(x)}{h} $$ 

whenever it exists and say that f is right-differentiable at x in such a case. Similarly, we denote by

 $$ \partial^{-}f(x)=\lim_{h\to0^{-}}\frac{f(x+h)-f(x)}{h} $$ 

whenever it exists and say that f is left-differentiable at x in such a case. Finally, we denote by

 $$ \partial f(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h} $$ 

whenever it exists and say that f is differentiable at x in such a case.

One-sided derivatives and classical derivatives are related by the following proposition.

Proposition B.3. Let  $ f : R \to R $  and  $ x \in R $ . f is differentiable at x if and only if f is right-differentiable at x and left-differentiable at x with  $ \partial^{+} f(x) = \partial^{-} f(x) $ . In such a case, we have  $ \partial f(x) = \partial^{+} f(x) = \partial^{-} f(x) $ .

### B.4 A CHAIN RULE FOR THE POSITIVE PART

The following proposition is concerned by the one-sided derivatives of a composition of functions involving the positive part.

Proposition B.4. Let  $ f : R \to R $  be a function that is continuous at 0 and define for all  $ h \in R $ ,  $ g(h) = [f(h)]^{+} $ .

• If f(0) < 0, then, g is differentiable at 0 and ∂g(0) = 0.

• If $f(0) = 0$ and $f$ is right-differentiable at 0, then, $g$ is right-differentiable at 0 and

$\partial^{+}g(0)=[\partial^{+}f(0)]^{+}$.

• If $f(0)=0$ and $f$ is left-differentiable at 0, then, $g$ is left-differentiable at 0 and $\partial^{-}g(0)=$

$-[-\partial^{-}f(0)]^{+}$.

• If $f(0) > 0$ and $f$ is right-differentiable at 0, then, $g$ is right-differentiable at 0 and $\partial^{+}g(0) = \partial^{+}f(0)$.

• If $f(0)>0$ and $f$ is left-differentiable at $0$, then, $g$ is left-differentiable at $0$ and $\partial^{-}g(0)=\partial^{-}f(0)$.

Proof.

• First, let us assume  $ f(0) < 0 $ , thus  $ g(0) = 0 $ . Since f is continuous at 0, there exists  $ \varepsilon > 0 $ , such that  $ f(h) < 0 $  for all  $ h \in (-\varepsilon, \varepsilon) $ . Therefore,  $ g(h) = [f(h)]^{+} = 0 $  for all  $ h \in (-\varepsilon, \varepsilon) $ . In particular,  $ \partial g(0) = \lim_{h \to 0} (g(h) - g(0))/h $  exists and is equal to 0.

• Now, assume  $  f(0) = 0  $ , thus  $  g(0) = 0  $ . Then for all h > 0, we have  $  g(h)/h = [f(h)]^{+}/h = [f(h)/h]^{+}  $  and since the positive part is a continuous function, we have  $  \lim_{h \to 0^{+}} g(h)/h = [\lim_{h \to 0^{+}} f(h)/h]^{+}  $  as soon as  $  \lim_{h \to 0^{+}} f(h)/h  $  exists. Similarly, for all h < 0 we have  $  g(h)/h = [f(h)]^{+}/h = -[-f(h)/h]^{+}  $  and since the positive part and  $ h \mapsto -h $  are continuous functions, we have  $  \lim_{h \to 0^{-}} g(h)/h = -[- \lim_{h \to 0^{-}} f(h)/h]^{+}  $  as soon as  $  \lim_{h \to 0^{-}} f(h)/h  $  exists.

• Finally, assume  $  f(0) > 0  $ . Since f is continuous at 0, there exists  $ \varepsilon > 0 $ , such that  $  f(h) > 0  $  for all  $  h \in (-\varepsilon, \varepsilon)  $ . Therefore,  $  g(h) = [f(h)]^{+} = f(h)  $  for all  $  h \in (-\varepsilon, \varepsilon)  $ . In particular, if  $  \lim_{h \to 0^{+}} (f(h) - f(0))/h  $  exists then  $  \lim_{h \to 0^{+}} (g(h) - g(0))/h  $  also exists and they are both equal. The same claim holds for  $  \lim_{h \to 0^{-}} (f(h) - f(0))/h  $  and  $  \lim_{h \to 0^{-}} (g(h) - g(0))/h  $ .

The following corollary gives a convenient way to compute one-sided derivatives of compositions involving the positive part.

Corollary B.5. Let  $ x \in R $  and  $ f : R \to R $  a function that is right-differentiable at x and left-differentiable at x. Then, the function g defined by  $ g(h) = [f(h)]^{+} $  for all  $ h \in R $ , is continuous at x, right-differentiable at x and left-differentiable at x, furthermore,

 $$ \partial^{+}g(x)=\left(\partial^{+}f(x)\right)\left(\mathbb{1}_{\left\{f(x)>0\right\}}+\mathbb{1}_{\left\{\partial^{+}f(x)>0\right\}}\mathbb{1}_{\left\{f(x)=0\right\}}\right), $$ 

 $$ \partial^{-}g(x)=\left(\partial^{-}f(x)\right)\left(\mathbb{1}_{\left\{f(x)>0\right\}}+\mathbb{1}_{\left\{\partial^{-}f(x)<0\right\}}\mathbb{1}_{\left\{f(x)=0\right\}}\right). $$ 

Proof. One can easily see that a function that is right-differentiable and left-differentiable at some point is necessarily continuous at this point. The rest of this corollary follows immediately from Proposition B.4. □

The following example considers the case of an affine function composed by the positive part. It will be used repeatedly in the following.

Example B.6. Let $a, b \in \mathbb{R}$ and define the function $g$ as $g(x) = [ax + b]^{+}$ for all $x \in \mathbb{R}$. Then, $g$ is left-differentiable and right-differentiable for all $x \in \mathbb{R}$ and,

 $$ \partial^{+}g(x)=a(\mathbb{1}_{\{a x+b>0\}}+\mathbb{1}_{\{a>0\}}\mathbb{1}_{\{a x+b=0\}}), $$ 

 $$ \partial^{-}g(x)=a(\mathbb{1}_{\{a x+b>0\}}+\mathbb{1}_{\{a<0\}}\mathbb{1}_{\{a x+b=0\}}). $$ 

Proof. This follows from Corollary B.5 when applied to  $ f(x) = ax + b $ .

### B.5 ONE-SIDED DERIVATIVES FOR OUR MODEL

Before providing the one-sided derivatives of each component of the model: the policy, the losses and the transitions, we recall the notations used.

• Time is indexed by $t \in \mathbb{N}$ and product types are indexed by $k \in [K]$.

• The states are denoted by  $ x_{t} = (x_{t,k,i})_{k \in [K], i \in [n_{k}]} \in \mathbb{R}^{n} $  with  $ n = \sum_{k=1}^{K} n_{k} $  and  $ n_{k} = m_{k} + L_{k} - 1 $  where  $ m_{k} $  and  $ L_{k} $  are respectively the lifetime and lead time of product k. The controls are denoted by  $ u_{t} = (u_{t,k})_{k \in [K]} \in \mathbb{R}^{K} $ . State-controls are denoted  $ z_{t} = (x_{t,k,1}, \ldots, x_{t,k,n_{k}}, u_{t,k})_{k \in [K]} \in \mathbb{R}^{n+K} $ .

• The parameters are denoted by $\theta_{t}=(\theta_{t,k,i})_{k\in[K],i\in[p_{k}]}\in\mathbb{R}^{P}$ and the features are denoted by $w_{t}=(w_{t,k,i})_{k\in[K],i\in[p_{k}]}\in\mathbb{R}^{P}$ where $P=\sum_{k=1}^{K}p_{k}$.

• All these quantities are related in our model through the transitions  $ f_{t} $  defined in Equation (5) and the policy  $ \pi_{t} $  defined in Equation (3.1).

• Finally, the loss functions  $ \ell_{t} $  are defined in Equation (6).

#### B.5.1 POLICY

The following proposition gives the formulas for the one-sided partial derivatives of the feature-enhanced base-stock policy.

Proposition B.7. Consider the feature-enhanced base-stock policy defined in Equation (3.1) and recalled here:

 $$ \pi_{t,k}(x_{t},\theta_{t})=\left[w_{t,k}^{\top}\theta_{t,k}-\sum_{i=1}^{n_{k}}x_{t,k,i}\right]^{+}. $$ 

The right partial derivatives of the policy are given by:

 $$ \begin{align*}\frac{\partial^{+}\pi_{t,k}(x_{t},\theta_{t})}{\partial x_{t,k^{\prime},i^{\prime}}}&=-\mathbb{1}_{\{k=k^{\prime}\}}\mathbb{1}_{\{w_{t,k}^{\top}\theta_{t,k}>\sum_{i=1}^{n_{k}}x_{t,k,i}\}},\\\frac{\partial^{+}\pi_{t,k}(x_{t},\theta_{t})}{\partial\theta_{t,k^{\prime},i^{\prime}}}&=w_{t,k,i^{\prime}}\mathbb{1}_{\{k=k^{\prime}\}}\big(\mathbb{1}_{\{w_{t,k}^{\top}\theta_{t,k}>\sum_{i=1}^{n_{k}}x_{t,k,i}\}}+\mathbb{1}_{\{w_{t,k,i^{\prime}}>0\}}\mathbb{1}_{\{w_{t,k}^{\top}\theta_{t,k}=\sum_{i=1}^{n_{k}}x_{t,k,i}\}}\big).\end{align*} $$ 

The left partial derivatives of the policy are given by:

 $$ \frac{\partial^{-}\pi_{t,k}(x_{t},\theta_{t})}{\partial x_{t,k^{\prime},i^{\prime}}}=-\mathbb{1}_{\{k=k^{\prime}\}}\mathbb{1}_{\{w_{t,k}^{\top}\theta_{t,k}\geq\sum_{i=1}^{n_{k}}x_{t,k,i}\}}, $$ 

 $$ \frac{\partial^{-}\pi_{t,k}(x_{t},\theta_{t})}{\partial\theta_{t,k^{\prime},i^{\prime}}}=w_{t,k,i^{\prime}}\mathbb{1}_{\{k=k^{\prime}\}}\big(\mathbb{1}_{\{w_{t,k}^{\top}\theta_{t,k}>\sum_{i=1}^{n_{k}}x_{t,k,i}\}}+\mathbb{1}_{\{w_{t,k,i^{\prime}}<0\}}\mathbb{1}_{\{w_{t,k}^{\top}\theta_{t,k}=\sum_{i=1}^{n_{k}}x_{t,k,i}\}}\big). $$ 

Proof. For each  $ t \in N $ ,  $ k, k' \in [K] $ ,  $ i \in [n_k] $ ,  $ i' \in [n_{k'}] $ , consider the functions:

 $$ x_{t,k^{\prime},i^{\prime}}\mapsto w_{t,k}^{\top}\theta_{t,k}-\sum_{i=1}^{n_{k}}x_{t,k,i}\mathrm{\quad and\quad }\theta_{t,k^{\prime},i^{\prime}}\mapsto w_{t,k}^{\top}\theta_{t,k}-\sum_{i=1}^{n_{k}}x_{t,k,i}. $$ 

These are univariate real-valued affine functions with respective derivatives:

 $$ \mathbf{-1}_{\{k=k^{\prime}\}}\mathrm{\boldmath~a n d~}\mathbf{1}_{\{k=k^{\prime}\}}w_{t,k,i^{\prime}}. $$ 

Applying Example B.6 for these two functions leads to the desired result.

#### B.5.2 STATE-CONTROL AFTER DISCARDING

Before moving to the derivations of one-sided partial derivatives of the losses and transitions which are more involved due to multiple compositions, we start by computing the derivatives of an auxiliary function: the state-control couple after discarding  $ \tilde{z}_{t} $  defined in Equation (4), with respect to the state-control couple  $ z_{t} $ .

Proposition B.8. Consider the vector the state-control couple after discarding  $ \tilde{z}_{t}\inR^{n+K} $  which definition is provided in Equation (4) and recalled here.

Given the state-control couple  $ z_{t} \in R^{n+K} $ , we have  $ \tilde{z}_{t,k,m_{k}} $  equal to,

 $$ \left[z_{t,k,m_{k}}-\frac{1}{v_{t,k}}\left[\left[\sum_{k^{\prime\prime}=1}^{K}\sum_{i^{\prime\prime}=1}^{m_{k^{\prime\prime}}}v_{t,k^{\prime\prime}}z_{t,k^{\prime\prime},i^{\prime\prime}}-V_{t}\right]^{+}-\sum_{k^{\prime\prime}=1}^{k-1}v_{t,k^{\prime\prime}}z_{t,k^{\prime\prime},m_{k^{\prime\prime}}}\right]^{+}\right]^{+}, $$ 

and  $ \tilde{z}_{t,k,i}=z_{t,k,i} $  for all  $ i\in[n_{k}+1]\setminus\{m_{k}\} $ 

If  $ i \in [n_{k} + 1] \setminus \{m_{k}\} $ , we have:

 $$ \frac{\partial\tilde{z}_{t,k,i}}{\partial z_{t,k^{\prime},i^{\prime}}}=\mathbb{1}_{\{k=k^{\prime}\}}\mathbb{1}_{\{i=i^{\prime}\}}. $$ 

Now assume $i = m_{k}$, then, the one-sided partial derivatives are given by:

 $$ \frac{\partial^{+}\tilde{z}_{t,k,i}}{\partial z_{t,k^{\prime},i^{\prime}}}=\partial^{+}\alpha\cdot(\mathbb{1}_{\{\alpha>0\}}+\mathbb{1}_{\{\partial^{+}\alpha>0\}}\mathbb{1}_{\{\alpha=0\}}), $$ 

 $$ \frac{\partial^{-}\tilde{z}_{t,k,i}}{\partial z_{t,k^{\prime},i^{\prime}}}=\partial^{-}\alpha\cdot(\mathbb{1}_{\{\alpha>0\}}+\mathbb{1}_{\{\partial^{-\alpha}>0\}}\mathbb{1}_{\{\alpha=0\}}), $$ 

where:

 $$ \alpha=z_{t,k,i}-\frac{1}{v_{t,k}}\left[\beta\right]^{+}, $$ 

 $$ \beta=\left[\sum_{k^{\prime\prime}=1}^{K}\sum_{i^{\prime\prime}=1}^{m_{k^{\prime\prime}}}v_{t,k^{\prime\prime}}z_{t,k^{\prime\prime},i^{\prime\prime}}-V_{t}\right]^{+}-\sum_{k^{\prime\prime}=1}^{k-1}v_{t,k^{\prime\prime}}z_{t,k^{\prime\prime},m_{k^{\prime\prime}}}, $$ 

 $$ \partial^{+}\alpha=\mathbb{1}_{\{k=k^{\prime}\}}\mathbb{1}_{\{i=i^{\prime}\}}-\frac{1}{v_{t,k}}\partial^{+}\beta\cdot\left(\mathbb{1}_{\{\beta>0\}}+\mathbb{1}_{\{\partial^{+}\beta>0\}}\mathbb{1}_{\{\beta=0\}}\right), $$ 

 $$ \begin{array}{r}{\partial^{+}\beta=v_{t,k^{\prime}}\big(\mathbb{1}_{\{i^{\prime}\in[m_{k^{\prime}}]\}}\mathbb{1}_{\{\sum_{k^{\prime\prime}=1}^{K}\sum_{i^{\prime\prime}=1}^{m_{k^{\prime\prime}}}v_{t,k^{\prime\prime}}z_{t,k^{\prime\prime},i^{\prime\prime}}\geq V_{t}\}}-\mathbb{1}_{\{k^{\prime}\in[k-1]\}}\mathbb{1}_{\{i^{\prime}=m_{k^{\prime}}\}}\big),}\end{array} $$ 

 $$ \partial^{-}\alpha=\mathbb{1}_{\{k=k^{\prime}\}}\mathbb{1}_{\{i=i^{\prime}\}}-\frac{1}{v_{t,k}}\partial^{-}\beta\cdot\left(\mathbb{1}_{\{\beta>0\}}+\mathbb{1}_{\{\partial^{-\beta}<0\}}\mathbb{1}_{\{\beta=0\}}\right), $$ 

 $$ \partial^{-}\beta=v_{t,k^{\prime}}\big(\mathbb{1}_{\{i^{\prime}\in[m_{k^{\prime}}]\}}\mathbb{1}_{\{\sum_{k^{\prime\prime}=1}^{K}\sum_{i^{\prime\prime}=1}^{m_{k^{\prime\prime}}}v_{t,k^{\prime\prime}}z_{t,k^{\prime\prime},i^{\prime\prime}}>V_{t}\}}-\mathbb{1}_{\{k^{\prime}\in[k-1]\}}\mathbb{1}_{\{i^{\prime}=m_{k^{\prime}}\}}\big). $$ 

Proof. The case  $ i \in [n_{k} + 1] \setminus \{m_{k}\} $  is trivial. On the other hand, the formulas for the case  $ i = m_{k} $  are obtained by applying repeatedly Corollary B.5. First, consider the univariate real-valued affine function:

 $$ z_{t,k^{\prime},i^{\prime}}\mapsto\sum_{k^{\prime\prime}=1}^{K}\sum_{i^{\prime\prime}=1}^{m_{k^{\prime\prime}}}v_{t,k^{\prime\prime}}z_{t,k^{\prime\prime},i^{\prime\prime}}-V_{t}. $$ 

and apply Corollary B.5 (or Example B.6). Then, we derive easily  $ \partial^{+}\beta $  and  $ \partial^{-}\beta $ . Applying Corollary B.5 to  $ \beta $ , leads us easily to  $ \partial^{+}\alpha $  and  $ \partial^{-}\alpha $ . A final application of Corollary B.5 to  $ \alpha $  allows us to conclude.

#### B.5.3 LOSSES

In the following we give, in the general case, the one-sided partial derivatives of the loss functions used of our model, defined in Equation (6).

Proposition B.9. Consider the loss functions of our model defined in Equation (6). The one-sided partial derivatives of the losses are given by the following, for each  $ \square\in\{-,+\} $ ,

 $$ \begin{aligned}\frac{\partial^{\square}\ell_{t}(z_{t})}{\partial z_{t,k^{\prime},i^{\prime}}}=\sum_{k=1}^{K}&\left(c_{t,k}^{\mathrm{pure}}\mathbb{1}_{\{k=k^{\prime}\}}\mathbb{1}_{\{i^{\prime}=m_{k^{\prime}}+L_{k^{\prime}}\}}\right.\\&\left.+c_{t,k}^{\mathrm{hold}}\partial^{\square}\gamma^{\mathrm{hold}}\cdot\left(\mathbb{1}_{\{\sum_{i=1}^{m_{k}}\tilde{z}_{t,k,i}>d_{t,k}\}}+\mathbb{1}_{\{\square\cdot\partial^{\square}\gamma^{\mathrm{hold}}>0\}}\mathbb{1}_{\{\sum_{i=1}^{m_{k}}\tilde{z}_{t,k,i}=d_{t,k}\}}\right)\right.\\&\left.-c_{t,k}^{\mathrm{pena}}\partial^{\square}\gamma^{\mathrm{hold}}\cdot\left(\mathbb{1}_{\{\sum_{i=1}^{m_{k}}\tilde{z}_{t,k,i}<d_{t,k}\}}+\mathbb{1}_{\{\square\cdot\partial^{\square}\gamma^{\mathrm{hold}}<0\}}\mathbb{1}_{\{\sum_{i=1}^{m_{k}}\tilde{z}_{t,k,i}=d_{t,k}\}}\right)\right.\\&\left.+c_{t,k}^{\mathrm{outd}}\partial^{\square}\gamma^{\mathrm{outd}}\cdot\left(\mathbb{1}_{\{\tilde{z}_{t,k,1}>d_{t,k}\}}+\mathbb{1}_{\{\square\cdot\partial^{\square}\gamma^{\mathrm{outd}}>0\}}\mathbb{1}_{\{\tilde{z}_{t,k,1}=d_{t,k}\}}\right)\right.\\&\left.+c_{t,k}^{\mathrm{over}}\cdot\left(\mathbb{1}_{\{k=k^{\prime}\}}\mathbb{1}_{\{i^{\prime}=m_{k^{\prime}}\}}-\frac{\partial^{\square}\tilde{z}_{t,k,m_{k}}}{\partial z_{t,k^{\prime},i^{\prime}}}\right)\right),\end{aligned} $$ 

where:

 $$ \partial^{\Box}\gamma^{\mathrm{h o l d}}=\frac{\partial^{\Box}\tilde{z}_{t,k,m_{k}}}{\partial z_{t,k^{\prime},i^{\prime}}}+\mathbb{1}_{\{k^{\prime}=k\}}\mathbb{1}_{\{i^{\prime}\in[m_{k^{\prime}}-1]\}}, $$ 

 $$ \partial^{\Box}\gamma^{\mathrm{o u t d}}=\mathbb{1}_{\{m_{k}=1\}}\frac{\partial^{\Box}\tilde{z}_{t,k,m_{k}}}{\partial z_{t,k^{\prime},i^{\prime}}}+\mathbb{1}_{\{m_{k}\neq1\}}\mathbb{1}_{\{k^{\prime}=k\}}\mathbb{1}_{\{i^{\prime}=m_{k^{\prime}}\}}, $$ 

and  $ \partial^{\Box}\tilde{z}_{t,k,m_{k}}/\partial z_{t,k^{\prime},i^{\prime}} $  is given by Proposition B.8.

Proof. This follows from Proposition B.8 and Corollary B.5.

#### B.5.4 TRANSITIONS

In the following we give, in the general case, the one-sided partial derivatives of the transition functions of our model, defined in Equation (5).

Proposition B.10. Consider the transition functions of our model defined in Equation (5) and recalled here:

 $$ f_{t,k,i}(z_{t})=\begin{cases}\left[\tilde{z}_{t,k,i+1}-\left[d_{t,k}-\sum_{i^{\prime}=1}^{i}\tilde{z}_{t,k,i^{\prime}}\right]^{+}\right]^{+}&for i=1,\cdots,m_{k}-1,\\\tilde{z}_{t,k,i+1}&for i=m_{k},\cdots,m_{k}+L_{k}-1.\end{cases} $$ 

where  $ \tilde{z}_{t} $  is the state-control couple after discarding defined in Equation (4).

If  $ i \in [n_{k}] \setminus \{m_{k} - 1\} $ , then, the one-sided derivatives of the transition functions are given by:

 $$ \frac{\partial^{+}f_{t,k,i}(z_{t})}{z_{t,k^{\prime},i^{\prime}}}=\mathbb{1}_{\{k=k^{\prime}\}}\Big(\mathbb{1}_{\{i\in[n_{k}]\backslash[m_{k}-1]\}}\mathbb{1}_{\{i^{\prime}=i+1\}}+\mathbb{1}_{\{i\in[m_{k}-1]\}}\Big( $$ 

 $$ \begin{array}{r}{\mathbb{1}_{\left\{i^{\prime}=i+1\right\}}\mathbb{1}_{\left\{z_{t,k,i+1}\geq\left[d_{t,k}-\sum_{i^{\prime\prime}=1}^{i}z_{t,k,i^{\prime\prime}}\right]^{+}\right\}}+\mathbb{1}_{\left\{i^{\prime}\in[i]\right\}}\mathbb{1}_{\left\{z_{t,k,i+1}\geq d_{t,k}-\sum_{i^{\prime\prime}=1}^{i}z_{t,k,i^{\prime\prime}}>0\right\}})\Biggr),}\end{array} $$ 

 $$ \frac{\partial^{-}f_{t,k,i}(z_{t})}{z_{t,k^{\prime},i^{\prime}}}=\mathbb{1}_{\{k=k^{\prime}\}}\Big(\mathbb{1}_{\{i\in[n_{k}]\backslash[m_{k}-1]\}}\mathbb{1}_{\{i^{\prime}=i+1\}}+\mathbb{1}_{\{i\in[m_{k}-1]\}}\Big( $$ 

 $$ \begin{array}{r}{\mathbb{1}_{\left\{i^{\prime}=i+1\right\}}\mathbb{1}_{\left\{z_{t,k,i+1}>\left[d_{t,k}-\sum_{i^{\prime\prime}=1}^{i}z_{t,k,i^{\prime\prime}}\right]^{+}\right\}}+\mathbb{1}_{\left\{i^{\prime}\in[i]\right\}}\mathbb{1}_{\left\{z_{t,k,i+1}>d_{t,k}-\sum_{i^{\prime\prime}=1}^{i}z_{t,k,i^{\prime\prime}}\geq0\right\}})\Biggr).}\end{array} $$ 

If  $ i = m_{k} - 1 $ , then, the one-sided derivatives of the transition functions are given by:

 $$ \frac{\partial^{+}f_{t,k,i}(z_{t})}{z_{t,k^{\prime},i^{\prime}}}=\partial^{+}\delta\cdot\left(\mathbb{1}_{\{\delta>0\}}+\mathbb{1}_{\{\partial^{+}\delta>0\}}\mathbb{1}_{\{\delta=0\}}\right) $$ 

 $$ \frac{\partial^{-}f_{t,k,i}(z_{t})}{z_{t,k^{\prime},i^{\prime}}}=\partial^{-}\delta\cdot\left(\mathbb{1}_{\{\delta>0\}}+\mathbb{1}_{\{\partial^{-}\delta<0\}}\mathbb{1}_{\{\delta=0\}}\right), $$ 

where.

 $$ \begin{align*}\delta&=\tilde{z}_{t,k,m_{k}}-\left[d_{t,k}-\sum_{i^{\prime\prime}=1}^{m_{k}-1}z_{t,k,i^{\prime\prime}}\right]^{+},\\\partial^{+}\delta&=\frac{\partial^{+}\tilde{z}_{t,k,m_{k}}}{\partial z_{t,k^{\prime},i^{\prime}}}+\mathbb{1}_{\left\{k^{\prime}=k\right\}}\mathbb{1}_{\left\{i^{\prime}\in\left[m_{k^{\prime}}-1\right]\right\}}\mathbb{1}_{\left\{d_{t,k}>\sum_{i^{\prime\prime}=1}^{m_{k}-1}z_{t,k,i^{\prime\prime}}\right\}},\\\partial^{-}\delta&=\frac{\partial^{-}\tilde{z}_{t,k,m_{k}}}{\partial z_{t,k^{\prime},i^{\prime}}}+\mathbb{1}_{\left\{k^{\prime}=k\right\}}\mathbb{1}_{\left\{i^{\prime}\in\left[m_{k^{\prime}}-1\right]\right\}}\mathbb{1}_{\left\{d_{t,k}\geq\sum_{i^{\prime\prime}=1}^{m_{k}-1}z_{t,k,i^{\prime\prime}}\right\}},\\{^{\prime}}\partial z_{t,k^{\prime},i^{\prime}}&and\partial^{-}\tilde{z}_{t,k,m_{k}}/\partial z_{t,k^{\prime},i^{\prime}}are given by Proposition B.8.\end{align*} $$ 

 $$ \partial^{+}\tilde{z}_{t,k,m_{k}}/\partial z_{t,k^{\prime},i^{\prime}} $$ 

Proof. To prove this proposition, we start by recalling that $\tilde{z}_{t,k,i}=z_{t,k,i}$ for all indexes $i\neq m_{k}$. Thus, we can rewrite transitions replacing the coordinates of $\tilde{z}_{t}$ by those of $z_{t}$ at every spot except in the case $i=m_{k}-1$ where $\tilde{z}_{t,k,m_{k}}$ appears through the term $\tilde{z}_{t,k,i+1}$. Transitions are rewritten as follows:

 $$ f_{t,k,i}(z_{t})=\begin{cases}\left[z_{t,k,i+1}-\left[d_{t,k}-\sum_{i^{\prime}=1}^{i}z_{t,k,i^{\prime}}\right]^{+}\right]^{+}&for i=1,\cdots,m_{k}-2,\\\left[\tilde{z}_{t,k,m_{k}}-\left[d_{t,k}-\sum_{i^{\prime}=1}^{m_{k}-1}z_{t,k,i^{\prime}}\right]^{+}\right]^{+}&for i=m_{k}-1,\\z_{t,k,i+1}&for i=m_{k},\cdots,m_{k}+L_{k}-1.\end{cases} $$ 

The case  $ i \geq m_{k} $  is trivial, the case  $ i \in [m_{k}-2] $  can be dealt with two applications of Corollary B.5 and for the final case  $ i = m_{k}-1 $  we start by differentiating  $ \delta $  using Proposition B.8 and Example B.6. This leads us to the expressions of  $ \partial^{+}\delta $  and  $ \partial^{-}\delta $  and a final application of Corollary B.5 leads to the desired result.

### B.6 CENSORED DEMAND CASE

Here, we assume there are no warehouse-capacity constraints $(V_{t} = +\infty)$ and show that one can compute all the partial derivatives required by GAPSI using censored demand information. In this case, instead of observing the true demands $d_{t}$, we only observe a sales vector $s_{t} = (s_{t,k,i})_{k \in [K], i \in [m_{k}]}$, defined in our model by:

 $$ s_{t,k,i}=\min\left\{\tilde{z}_{t,k,i}\;,\left[d_{t,k}-\sum_{j=1}^{i-1}\tilde{z}_{t,k,j}\right]^{+}\right\}, $$ 

where  $ \tilde{z}_{t} $  is the state-control couple after discarding defined in Equation (4). The policy is completely known when computing its derivatives thus we only consider losses and transitions, which depend on the demand.

Notice that when letting $V_{t}=+\infty$, there is no overflow, the state-control couple after discarding defined in Equation (4) is given by $\tilde{z}_{t,k,i}=[z_{t,k,i}]^{+}$ for $i=m_{k}$ and $\tilde{z}_{t,k,i}=z_{t,k,i}$ otherwise. Since the model is such that state-control couples remain in $\mathbb{R}_{+}^{n+K}$, we can safely assume $\tilde{z}_{t,k,i}=z_{t,k,i}$ for all $i\in[n_{k}+1]$ instead, which will make derivations simpler.

We start by stating a property regarding the sales vector and then proceed to the computation of the left derivatives of losses and transitions in terms of the sales vector.

#### B.6.1 PROPERTIES OF THE SALES VECTOR

Proposition B.11. Consider the sales vector  $ s_{t} $  defined in Equation (11). Then, we have for all  $ k \in [K] $ ,  $ i \in [m_{k}] $ .

 $$ \sum_{j=1}^{i}s_{t,k,j}=\min\left\{\sum_{j=1}^{i}\tilde{z}_{t,k,j},d_{t,k}\right\}, $$ 

Proof. This property can be shown by a simple induction of  $ i = 1, \ldots, m_{k} $ .

For i = 1, we clearly have:

 $$ \sum_{j=1}^{1}s_{t,k,j}=s_{t,k,1}\stackrel{(11)}{=}\min\left\{\tilde{z}_{t,k,1}\;,\left[d_{t,k}\right]^{+}\right\}\stackrel{d_{t,k}\geq0}{=}\min\left\{\sum_{j=1}^{1}\tilde{z}_{t,k,j}\;,\boldsymbol{d}_{t,k}\right\}. $$ 

Assuming Equation (12) holds for some  $ i \in [m_{k} - 1] $ , we have:

 $$ \begin{align*}\sum_{j=1}^{i+1}s_{t,k,j}&\overset{(12)}{=}s_{t,k,i+1}+\min\left\{\sum_{j=1}^{i}\tilde{z}_{t,k,j}\ ,d_{t,k}\right\}\\&\overset{(11)}{=}\min\left\{\tilde{z}_{t,k,i+1}\ ,\left[d_{t,k}-\sum_{j=1}^{i}\tilde{z}_{t,k,j}\right]^{+}\right\}+\min\left\{\sum_{j=1}^{i}\tilde{z}_{t,k,j}\ ,d_{t,k}\right\}.\end{align*} $$ 

Since the following relations hold:

 $$ \left[d_{t,k}-\sum_{j=1}^{i}\tilde{z}_{t,k,j}\right]^{+}+\min\left\{\sum_{j=1}^{i}\tilde{z}_{t,k,j}\;,\;d_{t,k}\right\}=d_{t,k}, $$ 

 $$ \tilde{z}_{t,k,i+1}+\min\left\{\sum_{j=1}^{i}\tilde{z}_{t,k,j}\ ,d_{t,k}\right\}=\min\left\{\sum_{j=1}^{i+1}\tilde{z}_{t,k,j}\ ,d_{t,k}+z_{t,k,i+1}\right\}, $$ 

we can further simplify the expression of  $ \sum_{j=1}^{i+1}s_{t,k,j} $ 

 $$ \sum_{j=1}^{i+1}s_{t,k,j}=\min\left\{\sum_{j=1}^{i+1}\tilde{z}_{t,k,j}\ ,d_{t,k}+z_{t,k,i+1}\ ,d_{t,k}\right\}\stackrel{z_{t,k,i+1}\geq0}{=}\min\left\{\sum_{j=1}^{i+1}\tilde{z}_{t,k,j}\ ,d_{t,k}\right\}. $$ 

This concludes the proof.

#### B.6.2 LOSSES

Proposition B.12. Consider our model without warehouse-capacity constraints  $ (V_{t} = +\infty) $ . Then, the loss function defined in Equation (6) can be rewritten as:

 $$ \begin{aligned}\ell_{t}(z_{t})=\sum_{k=1}^{K}\Bigg(&c_{t,k}^{\mathrm{purc}}\cdot z_{t,k,m_{k}+L_{k}}+c_{t,k}^{\mathrm{hold}}\cdot\Bigg[\sum_{i=1}^{m_{k}}z_{t,k,i}-d_{t,k}\Bigg]^{+}\\&+c_{t,k}^{\mathrm{pena}}\cdot\Bigg[d_{t,k}-\sum_{i=1}^{m_{k}}z_{t,k,i}\Bigg]^{+}+c_{t,k}^{\mathrm{outd}}\cdot[z_{t,k,1}-d_{t,k}]^{+}\Bigg).\end{aligned} $$ 

The left partial derivatives of the losses can be written as follows:

 $$ \begin{aligned}&\frac{\partial^{-}\ell_{t}(z_{t})}{\partial z_{t,k^{\prime},i^{\prime}}}=c_{t,k^{\prime}}^{\mathrm{p u r c}}\mathbb{1}_{\{i^{\prime}=m_{k^{\prime}}+L_{k^{\prime}}\}}+c_{t,k^{\prime}}^{\mathrm{h o l d}}\mathbb{1}_{\{i^{\prime}\in[m_{k^{\prime}}]\}}\mathbb{1}_{\{\sum_{i=1}^{m_{k^{\prime}}}z_{t,k^{\prime},i}>\sum_{i=1}^{m_{k^{\prime}}}s_{t,k^{\prime},i}\}}\\ &-c_{t,k^{\prime}}^{\mathrm{p e n a}}\mathbb{1}_{\{i^{\prime}\in[m_{k^{\prime}}]\}}\mathbb{1}_{\{\sum_{i=1}^{m_{k^{\prime}}}z_{t,k^{\prime},i}=\sum_{i=1}^{m_{k^{\prime}}}s_{t,k^{\prime},i}\}}+c_{t,k^{\prime}}^{\mathrm{o u t d}}\mathbb{1}_{\{i^{\prime}=1\}}\mathbb{1}_{\{z_{t,k^{\prime},1}>s_{t,k^{\prime},1}\}}.\\ \end{aligned} $$ 

where  $ s_{t} $  is the sales vector defined in Equation (11).

Proof. First, we need to compute the left partial derivatives of the loss functions in the case  $ V_{t} = +\infty $ . To do so, we can either do it from scratch by applying Corollary B.5 (or Example B.6), or use Proposition B.9 while replacing  $ \partial^{-}\tilde{z}_{t,k,m_{k}}/\partial z_{t,k^{\prime},i^{\prime}} $  from Proposition B.8 by  $ 1_{\{k=k^{\prime}\}}1_{\{i^{\prime}=m_{k^{\prime}}\}} $ .

Either way, we obtain the following left partial derivatives for the losses:

 $$ \begin{align*}\frac{\partial^{-}\ell_{t}(z_{t})}{\partial z_{t,k^{\prime},i^{\prime}}}&=c_{t,k^{\prime}}^{\mathrm{purc}}\mathbb{1}_{\{i^{\prime}=m_{k^{\prime}}+L_{k^{\prime}}\}}+c_{t,k^{\prime}}^{\mathrm{hold}}\mathbb{1}_{\{i^{\prime}\in[m_{k^{\prime}}]\}}\mathbb{1}_{\{\sum_{i=1}^{m_{k^{\prime}}}z_{t,k^{\prime},i}>d_{t,k^{\prime}}\}}\\&\quad-c_{t,k^{\prime}}^{\mathrm{pena}}\mathbb{1}_{\{i^{\prime}\in[m_{k^{\prime}}]\}}\mathbb{1}_{\{d_{t,k^{\prime}}\geq\sum_{i=1}^{m_{k^{\prime}}}z_{t,k^{\prime},i}\}}+c_{t,k^{\prime}}^{\mathrm{outd}}\mathbb{1}_{\{i^{\prime}=1\}}\mathbb{1}_{\{z_{t,k^{\prime},1}>d_{t,k^{\prime}}\}}.\end{align*} $$ 

Then, using the property (12) from Proposition B.11, we observe that:

 $$ \sum_{j=1}^{i}z_{t,k,j}>d_{t,k}\iff\sum_{j=1}^{i}z_{t,k,j}>\sum_{j=1}^{i}s_{t,k,j} $$ 

for all  $ k \in [K] $ ,  $ i \in [m_{k}] $ . Considering this equivalence for  $ i = m_{k} $  and i = 1 leads to the desired result. ☐

#### B.6.3 TRANSITIONS

Proposition B.13. Consider our model without warehouse-capacity constraints  $ (V_{t} = +\infty) $ . Then, the transition functions defined in Equation (5) can be rewritten as:

 $$ f_{t,k,i}(z_{t})=\begin{cases}\left[z_{t,k,i+1}-\left[d_{t,k}-\sum_{i^{\prime}=1}^{i}z_{t,k,i^{\prime}}\right]^{+}\right]^{+}&for i=1,\cdots,m_{k}-1,\\z_{t,k,i+1}&for i=m_{k},\cdots,m_{k}+L_{k}-1.\end{cases} $$ 

The left partial derivatives of the losses can be written as follows:

 $$ \begin{aligned}&\frac{\partial^{-}f_{t,k,i}(z_{t})}{z_{t,k^{\prime},i^{\prime}}}=\Big(\mathbb{1}_{\{i\in[n_{k}]\backslash[m_{k}-1]\}}\mathbb{1}_{\{i^{\prime}=i+1\}}+\mathbb{1}_{\{i\in[m_{k}-1]\}}\big(\mathbb{1}_{\{i^{\prime}=i+1\}}\mathbb{1}_{\{f_{t,k,i}(z_{t})>0\}}\\ &+\mathbb{1}_{\{i^{\prime}\in[i]\}}\mathbb{1}_{\{\sum_{i^{\prime\prime}=1}^{i+1}z_{t,k,j}>\sum_{i^{\prime\prime}=1}^{i+1}s_{t,k,j}\}}\mathbb{1}_{\{\sum_{i^{\prime\prime}=1}^{i}z_{t,k,j}=\sum_{i^{\prime\prime}=1}^{i}s_{t,k,j}\}}\Big)\Big)\mathbb{1}_{\{k=k^{\prime}\}}.\\ \end{aligned} $$ 

where  $ s_{t} $  is the sales vector defined in Equation (11).

Proof. As in the proof of Proposition B.12, we first need to compute the left partial derivatives of the transition functions in the case  $ V_{t} = +\infty $ . To do so, we can either do it from scratch by applying Corollary B.5 twice, or use Proposition B.10 while replacing  $ \partial^{-}\tilde{z}_{t,k,m_{k}}/\partial z_{t,k',i'} $  from Proposition B.8 by  $ 1_{\{k=k'\}}1_{\{i'=m_{k'}\}} $ .

Either way, we obtain the following left partial derivatives for the transitions:

 $$ \begin{aligned}&\frac{\partial^{-}f_{t,k,i}(z_{t})}{z_{t,k^{\prime},i^{\prime}}}=\mathbb{1}_{\{k=k^{\prime}\}}\Big(\mathbb{1}_{\{i\in[n_{k}]\backslash[m_{k}-1]\}}\mathbb{1}_{\{i^{\prime}=i+1\}}+\mathbb{1}_{\{i\in[m_{k}-1]\}}\Big(\\&\mathbb{1}_{\{i^{\prime}=i+1\}}\mathbb{1}_{\{z_{t,k,i+1}>\left[d_{t,k}-\sum_{i^{\prime\prime}=1}^{i}z_{t,k,i^{\prime\prime}}\right]^{+}\}}+\mathbb{1}_{\{i^{\prime}\in[i]\}}\mathbb{1}_{\{z_{t,k,i+1}>d_{t,k}-\sum_{i^{\prime\prime}=1}^{i}z_{t,k,i^{\prime\prime}}\geq0\}}\Big)\Big).\\ \end{aligned} $$ 

Then, we observe the following equivalences for all  $ i \in [m_{k} - 1] $ ,

 $$ \begin{aligned}z_{t,k,i+1}>\left[d_{t,k}-\sum_{i^{\prime\prime}=1}^{i}z_{t,k,i^{\prime\prime}}\right]^{+}&\iff f_{t,k,i}(z_{t})>0,\\z_{t,k,i+1}>d_{t,k}-\sum_{i^{\prime\prime}=1}^{i}z_{t,k,i^{\prime\prime}}&\overset{(12)}{\Longleftrightarrow}\sum_{i^{\prime\prime}=1}^{i+1}z_{t,k,i^{\prime\prime}}>\sum_{i^{\prime\prime}=1}^{i+1}s_{t,k,i^{\prime\prime}},\\d_{t,k}-\sum_{i^{\prime\prime}=1}^{i}z_{t,k,i^{\prime\prime}}\geq0&\overset{(12)}{\Longleftrightarrow}\sum_{i^{\prime\prime}=1}^{i}z_{t,k,i^{\prime\prime}}=\sum_{i^{\prime\prime}=1}^{i}s_{t,k,i^{\prime\prime}}.\end{aligned} $$ 

This concludes the proof.

## C NUMERICAL EXPERIMENTS

In this section, we give additional details on the experiments of the main paper (in Subsections C.1, C.2 and C.3). We also present three new experiments: a simulation with Poisson demands where we use GAPSI in a setup where we know the optimal policy (Subsection C.4), a study on the impact of lifetimes and lead times on the performance (Subsection C.5), and a large scale experiments where we test GAPSI when there are hundreds of products in the inventory (Subsection C.6).

We give the exact definitions of the metrics we computed to evaluate the performances of GAPSI: the lost sales percentage, discussed in the main text, and two additional metrics classical in inventory problems, that are, the outdated percentage and the ratio of losses. They are respectively defined by:

 $$ \begin{aligned}&100\times\frac{\sum_{t=1}^{T}\sum_{k=1}^{K}\left[d_{t,k}-\sum_{i=1}^{m}z_{t,k,i}\right]^{+}}{\sum_{t=1}^{T}\sum_{k=1}^{K}d_{t,k}},\\&100\times\frac{\sum_{t=1}^{T}\sum_{k=1}^{K}\left[x_{t,k,1}-d_{t,k}\right]^{+}}{\sum_{t=1}^{T}\sum_{k=1}^{K}u_{t,k}},\\&\frac{\sum_{t=1}^{T}\ell_{t}(x_{t},u_{t})}{\sum_{t=1}^{T}\ell_{t}(x_{t}(S_{T}^{*}),u_{t}(S_{T}^{*}))},\end{aligned} $$ 

where  $ (x_{t}(S_{T}^{*}), u_{t}(S_{T}^{*}))_{t\in[T]} $  is the trajectory associated to  $ S_{T}^{*} $ .

For illustration purposes, Figure 6 shows the total sales of the M5 dataset, over the whole horizon and on a specific window.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_217_594_1005_857.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_222_868_999_1250.jpg" alt="Image" width="63%" /></div>


<div style="text-align: center;">Figure 6: Total demand in the M5 dataset across all the T = 1969 time periods (top) and a zoom on the last time periods (bottom).</div>


#### C.1 ADDITIONAL RESULTS FOR SECTION 4.1

We give in Table 2 results on the three metrics corresponding to the experiment of Section 4.1. The last column corresponds to the numbers indicated in the legend of Figure 2.

<div style="text-align: center;">Table 2: Metrics of several algorithms including GAPSI with features, correspondent to Figure 2</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Lost sales</td><td style='text-align: center;'>Outdating</td><td style='text-align: center;'>Ratio of losses</td></tr><tr><td style='text-align: center;'>GAPSI without features</td><td style='text-align: center;'>1.02%</td><td style='text-align: center;'>0.09%</td><td style='text-align: center;'>0.952</td></tr><tr><td style='text-align: center;'>Best cyclic base-stock policy</td><td style='text-align: center;'>0.75%</td><td style='text-align: center;'>0.08%</td><td style='text-align: center;'>0.906</td></tr><tr><td style='text-align: center;'>GAPSI with features</td><td style='text-align: center;'>0.66%</td><td style='text-align: center;'>0.05%</td><td style='text-align: center;'>0.851</td></tr></table>

#### C.2 ADDITIONAL RESULTS FOR SECTION 4.2

We give in Table 3 the computation time corresponding to the experiment.

<div style="text-align: center;">Table 3: Computation time in seconds.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td colspan="3">M5</td><td style='text-align: center;'>Califrais</td></tr><tr><td style='text-align: center;'>Total</td><td style='text-align: center;'>Category</td><td style='text-align: center;'>Product</td><td style='text-align: center;'>Total</td></tr><tr><td style='text-align: center;'>Base-stock levels  $ \hat{d}_{t} $</td><td style='text-align: center;'>0.22</td><td style='text-align: center;'>0.22</td><td style='text-align: center;'>0.21</td><td style='text-align: center;'>0.08</td></tr><tr><td style='text-align: center;'>MPC</td><td style='text-align: center;'>12.38</td><td style='text-align: center;'>83.98</td><td style='text-align: center;'>34.80</td><td style='text-align: center;'>5.93</td></tr><tr><td style='text-align: center;'>GAPSI without forecasts</td><td style='text-align: center;'>0.67</td><td style='text-align: center;'>0.67</td><td style='text-align: center;'>0.68</td><td style='text-align: center;'>0.26</td></tr><tr><td style='text-align: center;'>GAPSI with forecasts</td><td style='text-align: center;'>0.69</td><td style='text-align: center;'>0.67</td><td style='text-align: center;'>0.68</td><td style='text-align: center;'>0.25</td></tr></table>

#### C.3 ADDITIONAL RESULT FOR SECTION 4.3

The normalized standard deviation is defined as:

 $$ \sqrt{\frac{1}{T}\sum_{t=1}^{T}\left(\frac{d_{t,k}}{\sum_{t^{\prime}=1}^{T}d_{t^{\prime},k}/T}-1\right)^{2}}. $$ 

As an illustration, we show in Figure 7 four examples of demands, corresponding to different levels of variability with respect to this metric. The top curve has lowest normalized standard deviation, while the bottom one has the largest. We can see that for large standard deviation, there are changes of regimes in the demand, with some long periods of zero.

### C.4 CLASSICAL PERISHABLE INVENTORY SYSTEMS

In this series of simulations, we evaluate GAPSI as an algorithm for learning stationary base-stock policies in the context of classical perishable inventory systems. More precisely, we consider the experimental setup of Bu et al. (2023, Subsection 7.1). In this setup, there is a single product K = 1, which is perishable with a lifetime of m = 3 periods and has no order lead time, L = 0, and no warehouse-capacity constraint  $ V_{t} = +\infty $ . Its demand is drawn independently across time periods from a Poisson distribution with mean 5. The losses include purchase costs, holding costs, penalty costs and outdating costs with time-invariant unit costs.

According to the simulations of Bu et al. (2023), in this setup, the best stationary base-stock policy computed with distributional knowledge performs very well with a relative optimality gap of at most 0.48%, where the optimal baseline taken into consideration is the following:

 $$  OPT=\inf_{\pi\in\mathbb{I}}\limsup_{T\to+\infty}\frac{1}{T}\sum_{t=1}^{T}\mathbb{E}\left[\ell_{t}(x_{t}(\pi),u_{t}(\pi))\right], $$ 

with  $ \mathbb{P}=\{(\pi_{t})_{t\in\mathbb{N}}|\pi_{t}:\mathbb{X}\to\mathbb{U}\text{measurable}\} $ , that is,  $ \Pi $  is the set of time-varying policies mapping a state  $ x_{t} $  to an order quantity  $ u_{t}=\pi_{t}(x_{t}) $  through a measurable map  $ \pi_{t} $ .

The results of the simulations are given in Table 4. Each line of this table corresponds to a set of time-invariant unit costs:  $ (c^{\mathrm{purc}}, c^{\mathrm{pena}}, c^{\mathrm{outd}}) $ , and the unit holding cost is fixed to  $ c^{hold} = 1 $ .

<div style="text-align: center;"><img src="imgs/img_in_chart_box_219_165_1004_427.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_222_433_1004_693.jpg" alt="Image" width="63%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_217_700_1005_958.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_216_965_1004_1224.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 7: Demands (black), GAPSI's learned base-stock  $ S_{t} $  (blue) and best stationary base-stock policy  $ S_{T}^{*} $  (red) in problems with demands' normalized standard deviation of 0.21, 0.96, 1.12 and 3.38 (top to bottom).</div>


Instead of considering the online performances of GAPSI, we measure the performances of base-stock policies with base-stock level learned by GAPSI using the technique of averaging. This is a standard approach when converting an online algorithm to a stochastic optimization algorithm, see online-to-batch conversion (Orabona, 2019, Chapter 3). More specifically, after running GAPSI with  $ \Theta = [0, 20] $ ,  $ w_{t} = 1 $ ,  $ \eta = 0.1 $  and B = 10 against a (training) demand sequence of length

10000, we compute, for each  $ T \in \{100, 1000, 5000, 10000\} $ , the average base-stock level learned  $ \bar{S}_{T} = \sum_{t=1}^{T} S_{t}/T $  and evaluate the expected long-term average loss of the stationary base-stock policy associated to  $ \bar{S}_{T} $  using 100 (test) demand sequences of length 10000.

<div style="text-align: center;">Table 4: Expected average test loss of GAPSI's learned stationary base-stock policies at different points of the learning process and different unit costs.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>T=100</td><td style='text-align: center;'>T=1000</td><td style='text-align: center;'>T=5000</td><td style='text-align: center;'>T=10000</td><td style='text-align: center;'>OPT</td></tr><tr><td style='text-align: center;'>(0,8,3)</td><td style='text-align: center;'>4.32 \pm 0.15</td><td style='text-align: center;'>4.22 \pm 0.05</td><td style='text-align: center;'>4.20 \pm 0.04</td><td style='text-align: center;'>4.19 \pm 0.05</td><td style='text-align: center;'>4.16</td></tr><tr><td style='text-align: center;'>(0,8,6)</td><td style='text-align: center;'>4.41 \pm 0.16</td><td style='text-align: center;'>4.27 \pm 0.05</td><td style='text-align: center;'>4.27 \pm 0.05</td><td style='text-align: center;'>4.26 \pm 0.04</td><td style='text-align: center;'>4.23</td></tr><tr><td style='text-align: center;'>(0,8,8)</td><td style='text-align: center;'>4.45 \pm 0.19</td><td style='text-align: center;'>4.31 \pm 0.05</td><td style='text-align: center;'>4.31 \pm 0.05</td><td style='text-align: center;'>4.31 \pm 0.04</td><td style='text-align: center;'>4.28</td></tr><tr><td style='text-align: center;'>(0,20,8)</td><td style='text-align: center;'>6.01 \pm 0.43</td><td style='text-align: center;'>5.58 \pm 0.08</td><td style='text-align: center;'>5.58 \pm 0.08</td><td style='text-align: center;'>5.57 \pm 0.08</td><td style='text-align: center;'>5.50</td></tr><tr><td style='text-align: center;'>(0,40,8)</td><td style='text-align: center;'>8.12 \pm 1.03</td><td style='text-align: center;'>6.61 \pm 0.11</td><td style='text-align: center;'>6.62 \pm 0.11</td><td style='text-align: center;'>6.62 \pm 0.12</td><td style='text-align: center;'>6.56</td></tr><tr><td style='text-align: center;'>(5,8,3)</td><td style='text-align: center;'>28.16 \pm 0.18</td><td style='text-align: center;'>28.03 \pm 0.13</td><td style='text-align: center;'>28.03 \pm 0.12</td><td style='text-align: center;'>27.99 \pm 0.13</td><td style='text-align: center;'>28.01</td></tr><tr><td style='text-align: center;'>(5,8,6)</td><td style='text-align: center;'>28.18 \pm 0.19</td><td style='text-align: center;'>28.05 \pm 0.12</td><td style='text-align: center;'>28.04 \pm 0.12</td><td style='text-align: center;'>28.02 \pm 0.13</td><td style='text-align: center;'>28.02</td></tr><tr><td style='text-align: center;'>(5,8,8)</td><td style='text-align: center;'>28.16 \pm 0.16</td><td style='text-align: center;'>28.04 \pm 0.12</td><td style='text-align: center;'>28.02 \pm 0.15</td><td style='text-align: center;'>28.04 \pm 0.12</td><td style='text-align: center;'>28.03</td></tr><tr><td style='text-align: center;'>(5,20,8)</td><td style='text-align: center;'>30.79 \pm 0.42</td><td style='text-align: center;'>30.30 \pm 0.15</td><td style='text-align: center;'>30.30 \pm 0.14</td><td style='text-align: center;'>30.30 \pm 0.15</td><td style='text-align: center;'>30.26</td></tr><tr><td style='text-align: center;'>(5,40,8)</td><td style='text-align: center;'>33.12 \pm 0.93</td><td style='text-align: center;'>31.65 \pm 0.16</td><td style='text-align: center;'>31.64 \pm 0.17</td><td style='text-align: center;'>31.63 \pm 0.18</td><td style='text-align: center;'>31.57</td></tr></table>

Using Table 4 we can compute the relative optimality gap which is at most 1.25% for T = 10000. This indicates that GAPSI can be used to learn almost optimal base-stock policies in classical perishable inventory systems.

### C.5 IMPACT OF THE LIFETIME AND THE LEAD TIME

In this experiment, we study the impact of the lead time and the lifetime on GAPSI's performances in a single-product lost sales FIFO perishable inventory system without warehouse-capacity constraints $(K=1, V_{t}=+\infty)$. The demand of the product is given by the total demand of the M5 dataset. We consider time-invariant unit costs: $c^{\mathrm{purc}}=1$, $c_{t}^{\mathrm{hold}}=1$, $c_{t}^{\mathrm{outd}}=1$ and $c_{t}^{\mathrm{pena}}=10$.

For each value of lifetime m and lead time L, we ran both the best stationary base-stock policy  $ S_{T}^{*} $  in hindsight of the demand realizations and GAPSI with a single and constant feature:  $ w_{t} = (L + 1) \max_{s \in [T]} d_{s} $ , learning rate scale parameter  $ \eta = 0.1 $ , buffer size B = 50 over  $ \Theta = [0, 1] $ . We chose this constant feature, instead of  $ w_{t} = 1 $  for instance, so that the parameters can lie in  $ [0, 1] $  and can be interpreted as a ratio of the maximum demand.

<div style="text-align: center;">Table 5: Lost sales percentage</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>L=0</td><td style='text-align: center;'>L=1</td><td style='text-align: center;'>L=7</td><td style='text-align: center;'>L=14</td></tr><tr><td style='text-align: center;'>m=2</td><td style='text-align: center;'>1.02%</td><td style='text-align: center;'>2.45%</td><td style='text-align: center;'>4.02%</td><td style='text-align: center;'>5.10%</td></tr><tr><td style='text-align: center;'>m=7</td><td style='text-align: center;'>1.02%</td><td style='text-align: center;'>2.42%</td><td style='text-align: center;'>4.94%</td><td style='text-align: center;'>7.28%</td></tr><tr><td style='text-align: center;'>m=30</td><td style='text-align: center;'>1.02%</td><td style='text-align: center;'>2.42%</td><td style='text-align: center;'>4.94%</td><td style='text-align: center;'>7.20%</td></tr></table>

<div style="text-align: center;">Table 6: Outdating percentage</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>L=0</td><td style='text-align: center;'>L=1</td><td style='text-align: center;'>L=7</td><td style='text-align: center;'>L=14</td></tr><tr><td style='text-align: center;'>m=2</td><td style='text-align: center;'>0.09%</td><td style='text-align: center;'>0.26%</td><td style='text-align: center;'>1.27%</td><td style='text-align: center;'>4.46%</td></tr><tr><td style='text-align: center;'>m=7</td><td style='text-align: center;'>0%</td><td style='text-align: center;'>0%</td><td style='text-align: center;'>0%</td><td style='text-align: center;'>0.92%</td></tr><tr><td style='text-align: center;'>m=30</td><td style='text-align: center;'>0%</td><td style='text-align: center;'>0%</td><td style='text-align: center;'>0%</td><td style='text-align: center;'>0%</td></tr></table>

The results are given in Tables 5, 6 and 7. We observe that in all scenarios, GAPSI outperforms the best stationary base-stock policy, meaning that GAPSI can achieve negative regret. Furthermore,

<div style="text-align: center;">Table 7: Ratio of losses</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>L=0</td><td style='text-align: center;'>L=1</td><td style='text-align: center;'>L=7</td><td style='text-align: center;'>L=14</td></tr><tr><td style='text-align: center;'>m=2</td><td style='text-align: center;'>0.952</td><td style='text-align: center;'>0.943</td><td style='text-align: center;'>0.853</td><td style='text-align: center;'>0.793</td></tr><tr><td style='text-align: center;'>m=7</td><td style='text-align: center;'>0.954</td><td style='text-align: center;'>0.951</td><td style='text-align: center;'>0.825</td><td style='text-align: center;'>0.958</td></tr><tr><td style='text-align: center;'>m=30</td><td style='text-align: center;'>0.954</td><td style='text-align: center;'>0.951</td><td style='text-align: center;'>0.826</td><td style='text-align: center;'>0.904</td></tr></table>

even in the most difficult settings with high lead time and low lifetime (upper right corner of the tables), GAPSI managed to keep the lost sales moderate: at most 7.28%, and the outdated percentage small: at most 4.46%.

### C.6 LARGE SCALE EXPERIMENTS

Here, we run large scale experiments on the whole M5 dataset at the product level  $ (K = 3049) $  and the proprietary Califrais dataset  $ (K = 299) $  which features perishable products. In the M5 dataset, lifetimes, lead times and costs has been set as usual to  $ m_{k} = 3 $ ,  $ L_{k} = 0 $ ,  $ c_{t,k}^{purc} = 1 $ ,  $ c_{t,k}^{hold} = 1 $ ,  $ c_{t,k}^{outd} = 1 $  and  $ c_{t,k}^{pena} = 10 $ . On the other hand, Califrais dataset already include lifetimes  $ m_{k} \in \{3, 4, 5, 6, 7, 10, 30\} $ , lead times  $ L_{k} \in \{1, 2, 3, 4\} $ , time-varying unit purchase costs and time-varying unit selling prices. Purchase costs  $ c_{t,k}^{purc} $ , holding costs  $ c_{t,k}^{hold} $  and outdating cost  $ c_{t,k}^{outd} $  has been set to these purchase costs provided and the penalty cost has been set to 10 times the selling prices provided. Let us mention that the Califrais dataset features more erratic demands compared to the M5 dataset and only T = 860 days of data. Indeed, the normalized standard deviation in the Califrais dataset lie between 0.82 and 29.33 compared to 0.21 and 3.38 in the M5 dataset. Figure 8 shows two demand sequences from the Califrais dataset. Half of the demand sequences have higher normalized standard deviation than those depicted in this figure.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_222_810_1001_1329.jpg" alt="Image" width="63%" /></div>


<div style="text-align: center;">Figure 8: Two demand sequences from the Califrais dataset with respectively lowest normalized std. 0.83 (top) and median normalized std. 6.98 (bottom).</div>


For both datasets, we did not consider volume constraints  $ (V_{t}=+\infty) $  and GAPSI is run with time-invariant features  $ w_{t,k}=(L_{k}+1)\max_{s\in[T]}d_{s,k} $  and parameters  $ \eta=0.1 $ , B=10 over  $ \Theta=[0,1]^{K} $ .

<div style="text-align: center;">Table 8: Large scale experiments</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Lost sales</td><td style='text-align: center;'>Outdating</td></tr><tr><td style='text-align: center;'>M5 (K = 3049)</td><td style='text-align: center;'>4.86%</td><td style='text-align: center;'>4.74%</td></tr><tr><td style='text-align: center;'>Califrais (K = 299)</td><td style='text-align: center;'>18.05%</td><td style='text-align: center;'>6.07%</td></tr></table>

The results are given in Table 8. Overall, the performances are better on the M5 dataset compared to the Califrais dataset even though the former contains more than 10 times the number of products of the latter. This is due to several factors. In Califrais' dataset, the time horizon is shorter  $ (T = 860) $ , the replenishment is not instantaneous and the demand is more erratic. As we have seen in Sections 4.3 and C.5, important lead times and variance impact negatively the performances of GAPSI. This experiment shows that the number of products is less an issue compared to the properties of these products (lifetimes, lead times, demands' variance...). We think that employing coordinate-wise learning processes through AdaGrad for hyper-rectangles learning rates is helping dealing with such high-dimensional problems.