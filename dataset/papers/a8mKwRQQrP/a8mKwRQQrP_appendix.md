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