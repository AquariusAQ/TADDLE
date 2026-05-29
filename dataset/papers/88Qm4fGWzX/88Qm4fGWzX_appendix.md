## APPENDIX

The Appendix is organized as follows:

• In Sec. A, we show more implementation details.

- In Sec. B, we show more details of the SWiG-Event benchmark and the process of quantitative evaluation and image retrieval.

- In Sec. C, we show the results for attribute generation during event customization.

- In Sec. D, we provide the discussion of our work’s limitations and potential negative societal impacts.

- In Sec. E, we show more qualitative comparison results of event customization on the Real-Event.

## A IMPLEMENTATION DETAILS

The denoising process was set with 50 steps. For entity switching path, for all blocks and layers containing the cross-attention module, we apply the cross-attention guidance during the first 10 steps. And apply the cross-attention regulation during the whole 50 steps. For event transferring path, we perform spatial feature injection for block and layer at  $ \{decoder block 1 : [layer 1]\} $  during the whole 50 steps. And perform self-attention injection for blocks and layers at  $ \{decoder block 1 : [layer 1, 2]\} $ , decoder block 2:  $ [layer 0, 1, 2] $ , decoder block 3:  $ [layer 0, 1, 2]\} $  during the first 25 steps. We set the classifier-free guidance scale to 15.0.

## B DETAILS OF SWiG-Event and process of image retrieval

As shown in Figure 7(a), each SWiG-Event sample consists of a reference image with labeled bounding boxes and masks for each reference entity, the nouns of each reference entity, and the event class. As shown in Figure 7(b), we constructed the target prompt as a list of reference entity nouns. The ControlNet takes the semantic map merged from the masks as the layout condition, and BoxDiff takes the bounding boxes with labeled entity nouns as the layout condition.

To compare the image retrieval performance, we retrieved the target image for its corresponding reference image across all the 100 reference images that have the same reference event class.

## C Attribute Generation Results

In this paper, we didn’t explicitly model the attributes during generation. However, as the results are shown in Figure 5(b), since we can generate extra content for background and style by giving

<div style="text-align: center;"><img src="imgs/img_in_image_box_214_261_350_399.jpg" alt="Image" width="11%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_413_162_553_300.jpg" alt="Image" width="11%" /></div>


① Spiderman ② red apple

<div style="text-align: center;"><img src="imgs/img_in_image_box_619_161_761_300.jpg" alt="Image" width="11%" /></div>


① Spiderman ② green apple

<div style="text-align: center;"><img src="imgs/img_in_image_box_826_161_965_299.jpg" alt="Image" width="11%" /></div>


① Spiderman ② crystal apple

Reference Image

<div style="text-align: center;"><img src="imgs/img_in_image_box_414_362_553_500.jpg" alt="Image" width="11%" /></div>


① old lady ② cake

<div style="text-align: center;"><img src="imgs/img_in_image_box_619_362_759_501.jpg" alt="Image" width="11%" /></div>


① blonde lady ② cake

<div style="text-align: center;"><img src="imgs/img_in_image_box_826_362_964_501.jpg" alt="Image" width="11%" /></div>


① noble lady ② cake

<div style="text-align: center;">Figure 8: The results of attribute generation during event customization.</div>


corresponding text descriptions, we thus tried to model the attributes by giving extra adjectives to the target prompt as an easy and natural exploration. Meanwhile, to ensure the accurate generation of the attributes, we applied the cross-attention guidance and regulation on each attribute using the mask of the entity they describe. As the results shown in Figure 8, our method successfully addresses the attributes of the corresponding entity (e.g., colors, materials, and ages). After all, while the attribute part is not the primary focus of our work, our approach shows potential and effectiveness in addressing it, and we would be happy to conduct further research in our future work.

## D LIMITATION AND POTENTIAL NEGATIVE SOCIETAL IMPACT

Limitations. The main limitation of FreeEvent lies in the complexity of events and the number of entities. The customization effect may be compromised when there are too many entities in an image, especially if they are too small. As the first work in this direction, we hope our method can unveil new possibilities for more complex customization and the generation of a greater number of richer, more diverse entities. Additionally, since our model is built on pretrained Stable Diffusion (SD) models, our performance depends on the generative capabilities of SD. This can lead to suboptimal results for entities that the current SD struggles with, such as human faces and hands.

Potential Negative Societal Impacts. Since FreeEvent can seamlessly integrate with subject customization methods to generate target entities based on user-specified subjects, this capability also raises the same concerns about the potential misuse of pretrained SD models for malicious applications (e.g., Deepfakes) involving real human figures. To address this, it is essential to implement robust safeguards and ethical guidelines, similar to the security measures and NSFW content detection mechanisms already present in existing diffusion models.

## E MORE QUALITATIVE COMPARISON RESULTS

We show more comparisons on Real-Event in Figure 9, Figure 10, Figure 11, Figure 12 and Figure 13. Specifically, we list them by the order of entity numbers. And we use different combinations of target entities for the same reference image to generate diverse target images.

<div style="text-align: center;"><img src="imgs/img_in_image_box_214_192_1013_1351.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 9: Comparison of Event Customization. Different colors and numbers show the associations between reference entities and their corresponding target prompts.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_213_189_1011_1347.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 10: Comparison of Event Customization. Different colors and numbers show the associations between reference entities and their corresponding target prompts.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_213_201_1009_1336.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 11: Comparison of Event Customization. Different colors and numbers show the associations between reference entities and their corresponding target prompts.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_211_192_1012_1352.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 12: Comparison of Event Customization. Different colors and numbers show the associations between reference entities and their corresponding target prompts.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_214_194_1009_1351.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 13: Comparison of Event Customization. Different colors and numbers show the associations between reference entities and their corresponding target prompts.</div>