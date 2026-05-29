I need to summarize the appendix of an AI paper by splitting it into the original subsections, keeping each section under 500 tokens, using concise and accurate language, and retaining key academic information.

Let me go through each subsection and create a concise summary:

### Appendix A: Inventory Problems

#### A.1 Complete Expression of Transition Functions and Losses
This section provides mathematical formulations for inventory system transitions and losses. It covers:
- Order lead times transition functions
- Multi-product systems with warehouse-capacity constraints (with examples from Shi et al. 2016 and the authors' model)
- The loss function with various cost components (penalty, holding, purchase, outdating, overage)

Key points to include:
- The transition functions for state evolution
- The warehouse capacity constraint formulation
- The comprehensive loss function with all cost components

#### A.2 Equivalent Expressions for Discarding
This section provides an alternative, more interpretable formulation for the discarding process when warehouse capacity is exceeded. It introduces:
- The volume overflow definition
- Recursive volume removal calculation
- Proposition showing equivalence to the original formulation

Key points:
- The interpretation of the discarding process
- The recursive removal mechanism
- The mathematical equivalence proof

### Appendix B: GAPSI

#### B.1 The GAPS Algorithm
This section explains the GAPS algorithm (from Lin et al., 2024) for solving online inventory problems:
- The main idea of using online gradient descent with approximated gradients
- Two approximations made to make the gradient computation tractable
- The implementation details and theoretical guarantees

Key points:
- The surrogate loss function definition
- The two approximations for tractable gradient computation
- The theoretical regret bound

#### B.2 On the Selection of Derivatives
This section discusses a problematic behavior of GAPSI when derivatives are not carefully selected:
- A proposition showing how using left derivatives can cause the base-stock level to get stuck at zero
- The explanation of why this happens (due to non-convexity of the transformed loss)
- The recommendation to use right derivatives instead

Key points:
- The problematic behavior with left derivatives
- The theoretical explanation
- The recommendation for right derivatives

#### B.3 Definitions and Properties of One-Sided Derivatives
This section provides formal definitions and properties of one-sided derivatives:
- Right and left derivatives
- Relationship between one-sided and standard derivatives
- Conditions for differentiability

Key points:
- Formal definitions of right/left derivatives
- The relationship between one-sided and standard derivatives

#### B.4 A Chain Rule for the Positive Part
This section presents a chain rule for functions involving the positive part (ReLU):
- Proposition on derivatives of compositions with positive parts
- Corollary for convenient computation of one-sided derivatives

Key points:
- The chain rule for positive part functions
- The convenient form for computation

#### B.5 One-Sided Derivatives for Our Model
This section provides the one-sided partial derivatives for the model components:
- Policy derivatives (feature-enhanced base-stock policy)
- State-control after discarding
- Loss function derivatives
- Transition function derivatives

Key points:
- The right and left partial derivatives for each model component
- The mathematical expressions for these derivatives

#### B.6 Censored Demand Case
This section addresses the case with censored demand information (no warehouse constraints):
- Properties of the sales vector
- Loss function derivatives in terms of sales vector
- Transition function derivatives in terms of sales vector

Key points:
- The censored demand scenario
- The rewritten loss and transition functions
- The derivatives expressed in terms of observed sales

### Appendix C: Numerical Experiments

#### C.1 Additional Results for Section 4.1
This section provides additional metrics for the experiment in Section 4.1:
- Lost sales percentage
- Outdating percentage
- Ratio of losses

Key points:
- The performance metrics for different algorithms
- The comparison between GAPSI with/without features and the best cyclic base-stock policy

#### C.2 Additional Results for Section 4.2
This section provides computation time results for the experiment in Section 4.2:
- Time taken for different algorithms on different datasets

Key points:
- The computational efficiency of GAPSI compared to other methods
- The time taken for different algorithms on different datasets

#### C.3 Additional Result for Section 4.3
This section provides additional details on the demand variability experiment:
- The definition of normalized standard deviation
- Examples of demands with different variability levels

Key points:
- The metric for demand variability
- Illustrative examples of demand patterns

#### C.4 Classical Perishable Inventory Systems
This section evaluates GAPSI in classical perishable inventory systems:
- The experimental setup (single product, Poisson demand)
- Performance comparison with optimal base-stock policies
- The online-to-batch conversion technique

Key points:
- The experimental setup and results
- The relative optimality gap achieved by GAPSI
- The convergence of learned policies to optimal

#### C.5 Impact of the Lifetime and the Lead Time
This section studies the impact of lifetime and lead time on GAPSI's performance:
- Performance metrics across different lifetime and lead time combinations
- Comparison with best stationary base-stock policy

Key points:
- How lifetime and lead time affect performance
- The robustness of GAPSI across different settings

#### C.6 Large Scale Experiments
This section presents large-scale experiments on real-world datasets:
- Results on M5 dataset (3049 products) and Califrais dataset (299 products)
- Comparison of performance metrics
- Analysis of factors affecting performance

Key points:
- Performance on large-scale datasets
- Comparison between datasets with different characteristics
- Insights on factors affecting performance (time horizon, demand variability, etc.)

Now I'll write concise summaries for each subsection, keeping each under 500 tokens.
</think>

### Appendix A.1 Complete Expression of Transition Functions and Losses
This section formalizes the transition functions and loss functions for perishable inventory systems with lead times and warehouse constraints. For order lead times, the transition function is defined piecewise for different state indices. For multi-product systems, it provides two warehouse-capacity constraint formulations: one for Shi et al. (2016) with non-perishable products (L=0), and the authors' model with perishable products (m_k, L_k > 0). The loss function incorporates five cost components: penalty (for unmet demand), holding (for excess inventory), purchase (for ordering), outdating (for expired products), and overage (for discarded products). The formulation handles warehouse constraints by discarding products in ascending order when capacity is exceeded, with the discarded quantity computed recursively.

### Appendix A.2 Equivalent Expressions for Discarding
This section provides an interpretable alternative to the discarding process in warehouse-capacity constrained systems. It defines volume overflow o_t and recursively computes volume to remove r_t,k for each product k. Proposition A.1 proves that the total volume removed up to product k equals the minimum of cumulative product volume and overflow. The state-control vector after discarding is then expressed as z̃_t,k,m_k = z_t,k,m_k - r_t,k/v_t,k. This formulation clarifies the discarding process: products are discarded in order from k=1 to K, removing up to the overflow volume from each product's arrival.

### Appendix B.1 The GAPS Algorithm
GAPS solves online inventory problems using gradient descent with approximated gradients. It defines surrogate losses L_t(θ) as the cumulative loss if policy θ were followed from period 1 to t. The algorithm makes two key approximations: (1) computing gradients along the current trajectory instead of the ideal trajectory, and (2) truncating historical dependence to B recent time steps. The gradient approximation procedure (Algorithm 2) computes the gradient recursively using chain rules. Theoretical guarantees (Corollary 3.4 of Lin et al., 2024) show O(√T) regret under suitable assumptions, meaning GAPSI's cumulative loss approaches the optimal policy's loss as √T.

### Appendix B.2 On the Selection of Derivatives
This section demonstrates a problematic behavior when using left derivatives in GAPSI. Proposition B.1 shows that with base-stock policies and left derivatives (or ReLU'(0)=0), the base-stock level can get stuck at zero, causing suboptimal performance. The issue arises because the transformed loss function is not necessarily convex, even if the original loss is convex. The authors recommend using right derivatives instead, which avoids this stationary behavior and leads to better convergence properties.

### Appendix B.3 Definitions and Properties of One-Sided Derivatives
This section formally defines one-sided derivatives: right derivative ∂⁺f(x) = lim_{h→0⁺} (f(x+h)-f(x))/h and left derivative ∂⁻f(x) = lim_{h→0⁻} (f(x+h)-f(x))/h. Proposition B.3 states that f is differentiable at x iff it's both right- and left-differentiable with equal derivatives. These definitions are essential for handling non-differentiable functions like the ReLU activation used in the policy.

### Appendix B.4 A Chain Rule for the Positive Part
This section provides a chain rule for functions involving the positive part (ReLU). Proposition B.4 specifies derivatives of g(h)=[f(h)]⁺ based on f(0) and f's differentiability. Corollary B.5 gives convenient expressions for right and left derivatives: ∂⁺g(x) = ∂⁺f(x)(1_{f(x)>0} + 1_{∂⁺f(x)>0}1_{f(x)=0}) and similarly for left derivatives. This is crucial for computing gradients of the policy, which uses the positive part.

### Appendix B.5 One-Sided Derivatives for Our Model
This section derives one-sided partial derivatives for all model components. For the feature-enhanced base-stock policy π_t,k = [w_t^Tθ_t,k - Σx_t,k,i]⁺, it provides right and left derivatives with respect to state and parameters. For state-control after discarding, it gives derivatives using the chain rule from Proposition B.4. For losses and transitions, it expresses derivatives in terms of the policy derivatives and state-control derivatives, using the chain rule repeatedly. These derivatives are essential for implementing GAPSI's gradient approximation.

### Appendix B.6 Censored Demand Case
This section addresses the censored demand scenario (V_t=∞) where only sales s_t are observed, not true demand d_t. Proposition B.11 shows Σ_{j=1}^i s_t,k,j = min{Σ_{j=1}^i z_t,k,j, d_t,k}. Propositions B.12 and B.13 rewrite loss and transition derivatives in terms of observed sales, eliminating the need for true demand. The left derivatives simplify to expressions involving sales and state variables, making GAPSI applicable with censored demand information.

### Appendix C.1 Additional Results for Section 4.1
This section provides additional metrics for the experiment in Section 4.1. Table 2 shows lost sales (0.66% for GAPSI with features vs 0.75% for best cyclic base-stock), outdating (0.05% vs 0.08%), and loss ratio (0.851 vs 0.906). GAPSI with features outperforms both GAPSI without features and the best cyclic base-stock policy across all metrics, demonstrating the value of feature-based learning.

### Appendix C.2 Additional Results for Section 4.2
This section presents computation times for the experiment in Section 4.2. Table 3 shows GAPSI with/without forecasts are significantly faster than MPC (12.38s vs 0.67s for M5 total time). GAPSI's computation time is nearly constant across categories and products, demonstrating scalability. The base-stock policy computation is extremely fast (0.22s), while MPC scales poorly with problem size.

### Appendix C.3 Additional Result for Section 4.3
This section defines normalized standard deviation as √(1/T Σ(d_t,k/μ - 1)²), where μ is the mean demand. Figure 7 shows four demand examples with varying standard deviations (0.21 to 3.38), illustrating how higher variability leads to more regime changes and zero-demand periods. This metric characterizes demand patterns that affect inventory policy performance.

### Appendix C.4 Classical Perishable Inventory Systems
This section evaluates GAPSI in classical perishable inventory systems (K=1, m=3, L=0, Poisson demand). Using online-to-batch conversion, GAPSI learns stationary base-stock policies. Table 4 shows GAPSI's learned policies achieve relative optimality gaps of at most 1.25% (vs 0.48% for the optimal baseline). The gap closes as learning progresses (T=10000), demonstrating GAPSI's ability to learn near-optimal policies in classical settings.

### Appendix C.5 Impact of the Lifetime and the Lead Time
This section studies how lifetime (m) and lead time (L) affect performance in a single-product system. Tables 5-7 show GAPSI outperforms the best stationary base-stock policy across all settings. For high L and low m (L=14, m=2), GAPSI achieves 5.10% lost sales and 4.46% outdating, while the best stationary policy performs worse. The results demonstrate GAPSI's robustness to varying inventory parameters.

### Appendix C.6 Large Scale Experiments
This section presents experiments on large-scale datasets: M5 (K=3049 products) and Califrais (K=299 products). Table 8 shows GAPSI achieves 4.86% lost sales and 4.74% outdating on M5, and 18.05% and 6.07% on Califrais. Despite having 10x more products, M5 performs better due to longer horizon (T=1969 vs 860), less erratic demand (normalized std. 0.21-3.38 vs 0.82-29.33), and no lead times. The results show product count is less important than demand characteristics for GAPSI's performance.