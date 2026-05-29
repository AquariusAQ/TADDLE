## APPENDIX

## A DEFINITIONS

In this section, we provide the definitions of the terms simple polygon and visibility graph.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Terms</td><td style='text-align: center;'>Definitions</td></tr><tr><td style='text-align: center;'>Simple Polygon</td><td style='text-align: center;'>Let  $ V = (v_1, \ldots, v_n) $  be an ordered set of n points on the plane. The location of point  $ v_i $  is specified by its coordinates  $ (x_i, y_i) $ . Let  $ e_i = (v_i, v_{i+1}) $  be the set of line segments obtained by connecting consecutive points in V in a cyclic manner. These line segments define a closed planar curve - the boundary of a polygon  $ P $ . The points  $ v_i $  are the vertices of  $ P $  and the segments  $ e_i $  are its sides. Two consecutive edges of a polygon share an end-point at a vertex. In a simple polygon, these are the only intersections between the edges. The edges do not intersect each other.</td></tr><tr><td style='text-align: center;'>Visibility Graph</td><td style='text-align: center;'>A simple polygon  $ P $  has a well-defined interior and an exterior separated by its boundary  $ \delta P $ . This separation allows us to define visibility: We will use the notation  $ x \in P $  to denote that  $ x $  lies either on the boundary or the interior of  $ P $ . We say that two points  $ x, y \in P $  see each other if and only if  $ \forall z \in [x, y], z \in P $ . In other words, the line segment  $ [xy] $  lies completely inside or on the boundary of  $ P $ . The visibility graph of  $ P $ , denoted  $ G(P) $  is a graph that is a vertex to vertex relation of  $ P $ . There is an edge between two vertices  $ u $  and  $ v $  if and only if  $ u $  and  $ v $  are visible to each other in  $ P $ .</td></tr></table>

<div style="text-align: center;">Table 3: Definitions</div>


## B QUANTITATIVE RESULTS

In this section, we present additional results on the evaluation of the SDF Diffusion model, a comparison of computational costs with the baseline, and the performance of the baseline models on the out-of-distribution test set.

### B.1 OUT-OF-DISTRIBUTION BASELINE PERFORMANCE

In this section, we provide the baseline performance on the out-of-distribution dataset for Visibility Reconstruction problem. Table 2 shows results of VisDiff while Table 4 - 8 shows the results of baselines on out-of-distribution dataset. A comparison of F-1 scores indicates that VisDiff performs significantly better than all the baselines on the out-of-distribution dataset.

### B.2 SDF Diffusion Evaluation

We evaluate the SDF Diffusion model by measuring the L2 error between the ground truth and the predicted SDF on both in-distribution and out-distribution test datasets for the Visibility Reconstruction problem. Table 9 shows the performance of the SDF diffusion model. Our diffusion model predicts high-quality SDFs with low L2 error, indicating its effectiveness in capturing the underlying relationship between the polygon and visibility graphs.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Metrics</td><td style='text-align: center;'>Accuracy  $ \uparrow $</td><td style='text-align: center;'>Precision  $ \uparrow $</td><td style='text-align: center;'>Recall  $ \uparrow $</td><td style='text-align: center;'>F1-Score  $ \uparrow $</td></tr><tr><td style='text-align: center;'>Spiral</td><td style='text-align: center;'>0.712</td><td style='text-align: center;'>0.602</td><td style='text-align: center;'>0.602</td><td style='text-align: center;'>0.6</td></tr><tr><td style='text-align: center;'>Terrain</td><td style='text-align: center;'>0.726</td><td style='text-align: center;'>0.487</td><td style='text-align: center;'>0.646</td><td style='text-align: center;'>0.552</td></tr><tr><td style='text-align: center;'>Convex Fan</td><td style='text-align: center;'>0.569</td><td style='text-align: center;'>0.584</td><td style='text-align: center;'>0.573</td><td style='text-align: center;'>0.574</td></tr><tr><td style='text-align: center;'>Anchor</td><td style='text-align: center;'>0.788</td><td style='text-align: center;'>0.89</td><td style='text-align: center;'>0.854</td><td style='text-align: center;'>0.868</td></tr><tr><td style='text-align: center;'>Star</td><td style='text-align: center;'>0.575</td><td style='text-align: center;'>0.558</td><td style='text-align: center;'>0.588</td><td style='text-align: center;'>0.568</td></tr></table>

<div style="text-align: center;">Table 4: Specific polygon types: Sequence Prediction Performance</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Metrics</td><td style='text-align: center;'>Accuracy  $ \uparrow $</td><td style='text-align: center;'>Precision  $ \uparrow $</td><td style='text-align: center;'>Recall  $ \uparrow $</td><td style='text-align: center;'>F1-Score  $ \uparrow $</td></tr><tr><td style='text-align: center;'>Spiral</td><td style='text-align: center;'>0.758</td><td style='text-align: center;'>0.797</td><td style='text-align: center;'>0.457</td><td style='text-align: center;'>0.573</td></tr><tr><td style='text-align: center;'>Terrain</td><td style='text-align: center;'>0.8</td><td style='text-align: center;'>0.653</td><td style='text-align: center;'>0.578</td><td style='text-align: center;'>0.602</td></tr><tr><td style='text-align: center;'>Convex Fan</td><td style='text-align: center;'>0.615</td><td style='text-align: center;'>0.721</td><td style='text-align: center;'>0.434</td><td style='text-align: center;'>0.527</td></tr><tr><td style='text-align: center;'>Anchor</td><td style='text-align: center;'>0.698</td><td style='text-align: center;'>0.929</td><td style='text-align: center;'>0.694</td><td style='text-align: center;'>0.791</td></tr><tr><td style='text-align: center;'>Star</td><td style='text-align: center;'>0.631</td><td style='text-align: center;'>0.681</td><td style='text-align: center;'>0.47</td><td style='text-align: center;'>0.545</td></tr></table>

<div style="text-align: center;">Table 5: Specific polygon types: Encoder-Decoder Performance</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Metrics</td><td style='text-align: center;'>Accuracy  $ \uparrow $</td><td style='text-align: center;'>Precision  $ \uparrow $</td><td style='text-align: center;'>Recall  $ \uparrow $</td><td style='text-align: center;'>F1-Score  $ \uparrow $</td></tr><tr><td style='text-align: center;'>Spiral</td><td style='text-align: center;'>0.779</td><td style='text-align: center;'>0.904</td><td style='text-align: center;'>0.438</td><td style='text-align: center;'>0.587</td></tr><tr><td style='text-align: center;'>Terrain</td><td style='text-align: center;'>0.85</td><td style='text-align: center;'>0.841</td><td style='text-align: center;'>0.541</td><td style='text-align: center;'>0.654</td></tr><tr><td style='text-align: center;'>Convex Fan</td><td style='text-align: center;'>0.612</td><td style='text-align: center;'>0.888</td><td style='text-align: center;'>0.283</td><td style='text-align: center;'>0.427</td></tr><tr><td style='text-align: center;'>Anchor</td><td style='text-align: center;'>0.452</td><td style='text-align: center;'>0.865</td><td style='text-align: center;'>0.419</td><td style='text-align: center;'>0.553</td></tr><tr><td style='text-align: center;'>Star</td><td style='text-align: center;'>0.64</td><td style='text-align: center;'>0.892</td><td style='text-align: center;'>0.296</td><td style='text-align: center;'>0.442</td></tr></table>

<div style="text-align: center;">Table 6: Specific polygon types: GNN Performance</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Metrics</td><td style='text-align: center;'>Accuracy  $ \uparrow $</td><td style='text-align: center;'>Precision  $ \uparrow $</td><td style='text-align: center;'>Recall  $ \uparrow $</td><td style='text-align: center;'>F1-Score  $ \uparrow $</td></tr><tr><td style='text-align: center;'>Spiral</td><td style='text-align: center;'>0.773</td><td style='text-align: center;'>0.795</td><td style='text-align: center;'>0.507</td><td style='text-align: center;'>0.613</td></tr><tr><td style='text-align: center;'>Terrain</td><td style='text-align: center;'>0.843</td><td style='text-align: center;'>0.79</td><td style='text-align: center;'>0.553</td><td style='text-align: center;'>0.646</td></tr><tr><td style='text-align: center;'>Convex Fan</td><td style='text-align: center;'>0.676</td><td style='text-align: center;'>0.755</td><td style='text-align: center;'>0.573</td><td style='text-align: center;'>0.634</td></tr><tr><td style='text-align: center;'>Anchor</td><td style='text-align: center;'>0.756</td><td style='text-align: center;'>0.887</td><td style='text-align: center;'>0.817</td><td style='text-align: center;'>0.849</td></tr><tr><td style='text-align: center;'>Star</td><td style='text-align: center;'>0.675</td><td style='text-align: center;'>0.775</td><td style='text-align: center;'>0.494</td><td style='text-align: center;'>0.585</td></tr></table>

<div style="text-align: center;">Table 7: Specific polygon types: Vertex Diffusion Performance</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Metrics</td><td style='text-align: center;'>Accuracy  $ \uparrow $</td><td style='text-align: center;'>Precision  $ \uparrow $</td><td style='text-align: center;'>Recall  $ \uparrow $</td><td style='text-align: center;'>F1-Score  $ \uparrow $</td></tr><tr><td style='text-align: center;'>Spiral</td><td style='text-align: center;'>0.757</td><td style='text-align: center;'>0.918</td><td style='text-align: center;'>0.36</td><td style='text-align: center;'>0.516</td></tr><tr><td style='text-align: center;'>Terrain</td><td style='text-align: center;'>0.851</td><td style='text-align: center;'>0.909</td><td style='text-align: center;'>0.478</td><td style='text-align: center;'>0.625</td></tr><tr><td style='text-align: center;'>Convex Fan</td><td style='text-align: center;'>0.612</td><td style='text-align: center;'>0.94</td><td style='text-align: center;'>0.263</td><td style='text-align: center;'>0.41</td></tr><tr><td style='text-align: center;'>Anchor</td><td style='text-align: center;'>0.298</td><td style='text-align: center;'>0.98</td><td style='text-align: center;'>0.176</td><td style='text-align: center;'>0.299</td></tr><tr><td style='text-align: center;'>Star</td><td style='text-align: center;'>0.641</td><td style='text-align: center;'>0.945</td><td style='text-align: center;'>0.275</td><td style='text-align: center;'>0.425</td></tr></table>

<div style="text-align: center;">Table 8: Specific polygon types: Nelder-Mead Optimization Performance</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Test Dataset</td><td style='text-align: center;'>L2 Error  $ \downarrow $</td></tr><tr><td style='text-align: center;'>In-Distribution</td><td style='text-align: center;'>0.071</td></tr><tr><td style='text-align: center;'>Out-Distribution: Spiral</td><td style='text-align: center;'>0.091</td></tr><tr><td style='text-align: center;'>Out-Distribution: Terrain</td><td style='text-align: center;'>0.091</td></tr><tr><td style='text-align: center;'>Out-Distribution: Convex Fan</td><td style='text-align: center;'>0.083</td></tr><tr><td style='text-align: center;'>Out-Distribution: Anchor</td><td style='text-align: center;'>0.158</td></tr><tr><td style='text-align: center;'>Out-Distribution: Star</td><td style='text-align: center;'>0.069</td></tr></table>

<div style="text-align: center;">Table 9: SDF Evaluation: The table shows the L2 error between the predicted SDF from the diffusion model and the ground truth SDF</div>


### B.3 COMPUTATIONAL COST COMPARISON

We compare the computational cost of our model with the baselines by evaluating the inference time per sample. The inference time of VisDiff is higher compared to baseline models. The increased inference time is because VisDiff performs the inference in two steps through the SDF while other baselines achieve it in a single step. Furthermore, GNN, Sequence Prediction, and Encoder-Decoder generate only one sample per visibility graph while VisDiff and vertex diffusion generate 50 samples per visibility graph.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Baselines</td><td style='text-align: center;'>Computational Time (seconds)  $ \downarrow $</td></tr><tr><td style='text-align: center;'>Encoder-Decoder</td><td style='text-align: center;'>0.001</td></tr><tr><td style='text-align: center;'>Sequence Prediction</td><td style='text-align: center;'>0.075</td></tr><tr><td style='text-align: center;'>GNN</td><td style='text-align: center;'>0.005</td></tr><tr><td style='text-align: center;'>Optimization</td><td style='text-align: center;'>74.210</td></tr><tr><td style='text-align: center;'>Vertex Diffusion</td><td style='text-align: center;'>0.094</td></tr><tr><td style='text-align: center;'>VisDiff</td><td style='text-align: center;'>1.02</td></tr><tr><td style='text-align: center;'>Variational Autoencoder</td><td style='text-align: center;'>0.003</td></tr></table>

<div style="text-align: center;">Table 10: Computational Cost Comparison: Each inference time corresponds to the time in seconds taken for each model to generate vertex locations for a single visibility graph</div>


## C ABLATION STUDIES

The two main directions of ablation studies performed for VisDiff are in loss functions and architecture choices. Table 11 shows the results achieved for different architecture choices. It shows that the best results are achieved by estimating the SDF and vertex locations separately. We also evaluate the change in performance with an addition of each component of loss. Table 12 shows that with the addition of all the loss components helps gain 10% F1-Score than using just the vertex locations error.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Accuracy  $ \uparrow $</td><td style='text-align: center;'>Precision  $ \uparrow $</td><td style='text-align: center;'>Recall  $ \uparrow $</td><td style='text-align: center;'>F1  $ \uparrow $</td></tr><tr><td style='text-align: center;'>b) Joint</td><td style='text-align: center;'>0.78</td><td style='text-align: center;'>0.80</td><td style='text-align: center;'>0.60</td><td style='text-align: center;'>0.68</td></tr><tr><td style='text-align: center;'>d) Separate</td><td style='text-align: center;'>0.85</td><td style='text-align: center;'>0.83</td><td style='text-align: center;'>0.77</td><td style='text-align: center;'>0.80</td></tr></table>

<div style="text-align: center;">Table 11: Ablation Studies: (a) Joint estimation of SDF with vertex locations, (b) Separate estimation SDF with vertex locations (VisDiff). The results are on Visibility Reconstruction problem</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Accuracy  $ \uparrow $</td><td style='text-align: center;'>Precision  $ \uparrow $</td><td style='text-align: center;'>Recall  $ \uparrow $</td><td style='text-align: center;'>F1  $ \uparrow $</td></tr><tr><td style='text-align: center;'>a)  $ L_{MSE} $</td><td style='text-align: center;'>0.83</td><td style='text-align: center;'>0.74</td><td style='text-align: center;'>0.72</td><td style='text-align: center;'>0.73</td></tr><tr><td style='text-align: center;'>b)  $ L_{MSE} + L_{Vis} $</td><td style='text-align: center;'>0.84</td><td style='text-align: center;'>0.83</td><td style='text-align: center;'>0.70</td><td style='text-align: center;'>0.76</td></tr><tr><td style='text-align: center;'>c)  $ L_{MSE} + L_{Vis} + L_{Val} $</td><td style='text-align: center;'>0.85</td><td style='text-align: center;'>0.83</td><td style='text-align: center;'>0.73</td><td style='text-align: center;'>0.78</td></tr><tr><td style='text-align: center;'>d)  $ L_{MSE} + L_{Vis} + L_{Val} + L_{SDF} $</td><td style='text-align: center;'>0.85</td><td style='text-align: center;'>0.83</td><td style='text-align: center;'>0.77</td><td style='text-align: center;'>0.80</td></tr></table>

<div style="text-align: center;">Table 12: Ablation Studies Loss Components: (a) MSE Loss, (b) Adding Visibility Loss component, (c) Adding Visibility and Validity Loss component, (d) Adding Visibility, Validity and SDF Loss component. The results are on Visibility Reconstruction problem</div>


## D DATASET STATISTICS

In this section, we present statistics about our dataset. Figure 8 shows the distribution of the train and in-distribution test set statistics. It shows that our dataset is uniform in diameter of the visibility

graph. Figure 9 compares the training dataset with the out-of-distribution testing dataset. It shows that star, convex-fan, and terrain classes have densities different from our train distribution, where density refers to the percentage of edges in the visibility graph.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_236_296_577_562.jpg" alt="Image" width="27%" /></div>


<div style="text-align: center;">(a) Density Comparison Train vs Test</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_618_297_971_561.jpg" alt="Image" width="28%" /></div>


<div style="text-align: center;">(b) Diameter Comparison Train vs Test</div>


<div style="text-align: center;">Figure 8: Train vs in-distribution test set analysis: 8a) The density is inversely proportional to the diameter. Uniform sampling of diameter results in bimodal density. 8b) Training and testing sets are uniform in terms of the link diameter of the visibility graph.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_223_766_577_1030.jpg" alt="Image" width="28%" /></div>


<div style="text-align: center;">(a) Density Comparison Train vs Test</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_618_766_971_1030.jpg" alt="Image" width="28%" /></div>


<div style="text-align: center;">(b) Diameter Comparison Train vs Test</div>


<div style="text-align: center;">Figure 9: Out-of-distribution test set analysis: Figure 9a shows the density of the anchor and spiral are close to the mean of the bimodal training distribution, making it similar to our training set. The density of the star, convex fan, and terrain differ significantly from the training distribution.</div>


## E QUALITATIVE RESULTS

In this section, we provide additional qualitative results on Visibility Reconstruction, Visibility Characterization, Visibility Recognition, and the Triangulation problem (Section 6.2.4).

### E.1 Visibility Reconstruction

We provide additional qualitative results for the Visibility Reconstruction problem. Figures 10 and 11 show the comparison between polygons generated by VisDiff to baselines. The F1-Score shows that VisDiff generates polygons much closer to the visibility graph of the ground truth polygon.

<div style="text-align: center;"><img src="imgs/img_in_image_box_241_159_338_243.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_346_159_441_243.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_242_247_338_335.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(a) GT</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_346_247_441_336.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(b) 0.73</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_451_160_548_335.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(c) 0.65</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_560_160_658_335.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;">(d) 0.55</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_669_158_765_244.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_670_252_765_336.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(e) 0.54</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_778_158_872_242.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_778_250_873_336.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(f) 0.54</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_886_159_978_239.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_886_250_980_335.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(g) 0.49</div>


Figure 10: Visibility reconstruction qualitative results: The top row shows the polygons generated by different methods. The first vertex is represented by deep purple and the last vertex by yellow (anticlockwise ordering). The second row shows corresponding visibility graphs of the polygons where green represents the visible edge and red represents the non-visible edge. The polygon results correspond to the following methods - a) Ground Truth, b) VisDiff c) Sequence Prediction d) GNN, e) Vertex diffusion, f) Encoder-Decoder, g) Optimization. The captions indicate F1-Score

<div style="text-align: center;"><img src="imgs/img_in_image_box_243_549_337_629.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_242_637_338_723.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(a) GT</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_344_550_438_629.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_345_637_440_724.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(b) 0.80</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_452_549_548_724.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(c) 0.71</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_560_548_657_724.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(d) 0.62</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_669_548_764_724.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(e) 0.54</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_778_548_872_724.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(f) 0.56</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_885_549_981_634.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_886_638_980_723.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(g) 0.50</div>


Figure 11: Visibility reconstruction qualitative results: The top row shows the polygons generated by different methods. The first vertex is represented by deep purple and the last vertex by yellow (anticlockwise ordering). The second row shows corresponding visibility graphs of the polygons where green represents the visible edge and red represents the non-visible edge. The polygon results correspond to the following methods - a) Ground Truth, b) VisDiff c) Sequence Prediction d) GNN, e) Vertex diffusion, f) Encoder-Decoder, g) Optimization. The captions indicate F1-Score

### E.2 Visibility Characterization

We provide further qualitative results on the problem of Visibility Characterization where we seek to generate the set of all polygons associated with the same visibility graph. Figures 12 and Figure 13 show the ability of VisDiff to sample multiple polygons given same visibility graph.

<div style="text-align: center;"><img src="imgs/img_in_image_box_219_1102_335_1181.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_437_1101_556_1180.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_660_1101_780_1179.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_890_1101_996_1171.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_214_1195_336_1276.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;">(a) GT</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_435_1196_557_1276.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;">(b) 0.77</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_661_1194_782_1276.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;">(c) 0.76</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_888_1196_1007_1276.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;">(d) 0.76</div>


Figure 12: Visibility Characterization: The top row shows multiple polygons generated by VisDiff for the same visibility graph G. The first vertex starts at deep purple and the last vertex ends at yellow (anticlockwise ordering). The second row shows the visibility graph corresponding to the polygons where green represents visible edge and red represents non-visible edge. Subfigure captions indicate the F1-Score

<div style="text-align: center;"><img src="imgs/img_in_image_box_213_158_316_231.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_445_158_547_231.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_662_158_776_238.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_214_250_336_335.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_435_248_557_335.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;">(a) GT</div>


<div style="text-align: center;">(b) 0.76</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_661_250_783_335.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_891_160_1008_242.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;">(c) 0.75</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_888_246_1008_335.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;">(d) 0.80</div>


<div style="text-align: center;">Figure 13: Visibility Characterization: The top row shows multiple polygons generated by VisDiff for the same visibility graph G. The first vertex is represented by deep purple and the last vertex by yellow (anticlockwise ordering). The second row shows the visibility graph corresponding to the polygons where green represents visible edge and red represents non-visible edge. Subfigure captions indicate the F1-Score</div>


### E.3 TRIANGULATION

We provide qualitative results for the problem of generating polygons from the triangulation. Figure 14 shows the performance of VisDiff compared to other baselines. VisDiff maintains 98% of the triangulation edges.

<div style="text-align: center;"><img src="imgs/img_in_image_box_242_664_339_736.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_346_664_441_738.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_242_746_338_840.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(a) GT</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_346_744_441_841.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(b) 0.98</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_453_664_549_737.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_452_744_548_841.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_560_663_659_736.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;">(c) 0.76</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_561_746_657_841.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_668_663_766_736.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;">(d) 0.92</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_669_747_765_841.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_776_663_872_736.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(e) 0.91</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_777_748_872_841.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_885_664_981_736.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(f) 0.81</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_886_747_980_840.jpg" alt="Image" width="7%" /></div>


<div style="text-align: center;">(g) 0.69</div>


<div style="text-align: center;">Figure 14: Triangulation Qualitative Results: Top row shows the polygons generated by different methods. The first vertex is represented by deep purple and the last vertex by yellow (anticlockwise ordering). The second row shows corresponding triangulation graphs of the polygons where green represents the triangulation edge and red represents the absence of the triangulation edge. The captions indicate the F1 Score of the triangulation graph compared to the GT. The polygon results correspond to the following methods - a) Ground Truth, b) VisDiff c) Sequence Prediction d) GNN, e) Vertex diffusion, f) Encoder-Decoder, g) Optimization</div>


### E.4 Visibility Recognition

We provide additional qualitative results to showcase failure and successful instances of VisDiff on Visibility Recognition problem. Figure 15 shows the output of VisDiff when the input is not a valid polygon (We generate visibility graphs of polygons with holes as invalid input samples). It shows that VisDiff can be used to identify non-valid visibility graphs in most of the scenarios by turning it into a classifier based on the validity of the output.

## F VARIATIONAL AUTOENCODER

In this section, we present the results for the variational autoencoder baseline. Specifically, we include out-of-distribution dataset results for visibility reconstruction, along with qualitative results for both the visibility reconstruction and triangulation problems. Table 13 reports the out-of-distribution test set performance of the variational autoencoder. Figure 16 shows the qualitative comparison of the variational autoencoder and VisDiff on the Visibility Reconstruction task. Figure 17 illustrates a qualitative comparison for the Triangulation problem.

<div style="text-align: center;"><img src="imgs/img_in_image_box_212_201_343_304.jpg" alt="Image" width="10%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_434_201_570_313.jpg" alt="Image" width="11%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_227_316_351_437.jpg" alt="Image" width="10%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_437_324_572_438.jpg" alt="Image" width="11%" /></div>


<div style="text-align: center;">(a)</div>


<div style="text-align: center;">(b)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_657_199_797_438.jpg" alt="Image" width="11%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_886_199_1022_305.jpg" alt="Image" width="11%" /></div>


<div style="text-align: center;">(c)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_886_321_1021_436.jpg" alt="Image" width="11%" /></div>


<div style="text-align: center;">(d)</div>


<div style="text-align: center;">Figure 15: Visibility Recognition: The top row signifies the ground truth non-valid polygon with the hole (red) while the bottom row is the polygons drawn by VisDiff. The first vertex is represented by deep purple and the last vertex by yellow (anticlockwise ordering). a) Non-Valid Sample 1: VisDiff predicts it as a non-valid polygon as it is not able to generate any valid polygon, b) Non-Valid Sample 2: VisDiff generates valid polygon where it learns to put points in a V shape to account for a hole. It misclassified a non-valid visibility graph as a valid visibility graph. c) Non-Valid Sample 3: VisDiff predicts it as a non-valid polygon as it is not able to generate any valid polygon, d) Non-Valid Sample 4: VisDiff predicts it as a non-valid polygon as it is not able to generate any valid polygon.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Metrics</td><td style='text-align: center;'>Accuracy  $ \uparrow $</td><td style='text-align: center;'>Precision  $ \uparrow $</td><td style='text-align: center;'>Recall  $ \uparrow $</td><td style='text-align: center;'>F1-Score  $ \uparrow $</td></tr><tr><td style='text-align: center;'>Spiral</td><td style='text-align: center;'>0.702</td><td style='text-align: center;'>0.602</td><td style='text-align: center;'>0.592</td><td style='text-align: center;'>0.6</td></tr><tr><td style='text-align: center;'>Terrain</td><td style='text-align: center;'>0.706</td><td style='text-align: center;'>0.467</td><td style='text-align: center;'>0.626</td><td style='text-align: center;'>0.534</td></tr><tr><td style='text-align: center;'>Convex Fan</td><td style='text-align: center;'>0.549</td><td style='text-align: center;'>0.564</td><td style='text-align: center;'>0.543</td><td style='text-align: center;'>0.553</td></tr><tr><td style='text-align: center;'>Anchor</td><td style='text-align: center;'>0.768</td><td style='text-align: center;'>0.87</td><td style='text-align: center;'>0.834</td><td style='text-align: center;'>0.851</td></tr><tr><td style='text-align: center;'>Star</td><td style='text-align: center;'>0.535</td><td style='text-align: center;'>0.538</td><td style='text-align: center;'>0.568</td><td style='text-align: center;'>0.552</td></tr></table>

<div style="text-align: center;">Table 13: Specific polygon types: Variational Autoencoder Performance</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_266_1057_429_1133.jpg" alt="Image" width="13%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_530_1057_687_1133.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_806_1056_943_1131.jpg" alt="Image" width="11%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_267_1141_429_1234.jpg" alt="Image" width="13%" /></div>


<div style="text-align: center;">(a) GT</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_533_1141_689_1234.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;">(b) 0.81</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_794_1150_955_1234.jpg" alt="Image" width="13%" /></div>


<div style="text-align: center;">(c) 0.64</div>


<div style="text-align: center;">Figure 16: Visibility reconstruction qualitative results: The top row shows the polygons generated by different methods. The first vertex is represented by deep purple and the last vertex by yellow (anticlockwise ordering). The second row shows corresponding visibility graphs of the polygons where green represents the visible edge and red represents the non-visible edge. The captions indicate the F1 Score of the visibility graph compared to the GT. The polygon results correspond to the following methods - a) Ground Truth, b) VisDiff, c) Variational Autoencoder</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_268_619_428_694.jpg" alt="Image" width="13%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_267_702_428_796.jpg" alt="Image" width="13%" /></div>


<div style="text-align: center;">(a) GT</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_528_619_690_796.jpg" alt="Image" width="13%" /></div>


<div style="text-align: center;">(b) 0.98</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_799_619_952_702.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_795_713_955_796.jpg" alt="Image" width="13%" /></div>


<div style="text-align: center;">(c) 0.75</div>


<div style="text-align: center;">Figure 17: Triangulation Qualitative Results: Top row shows the polygons generated by different methods. The first vertex is represented by deep purple and the last vertex by yellow (anticlockwise ordering). The second row shows corresponding triangulation graphs of the polygons where green represents the triangulation edge and red represents the absence of the triangulation edge. The captions indicate the F1 Score of the triangulation graph compared to the GT. The polygon results correspond to the following methods - a) Ground Truth, b) VisDiff, c) Variational Autoencoder</div>