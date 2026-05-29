## 8 APPENDIX

### 8.1 DETAILS OF ICON-DESCRIPTION DATASET

In figure 5, we see that the original BLIP-2 model tend to focus on describing shapes and colors of app icons, while struggling to recognize the semantics of the icon. This motivates us to finetune this model on an icon description dataset. For the dataset, we use the result of parsed icon bounding boxes inferenced by the interactable icon detection model on the ScreenSpot dataset since it contains screenshots on both mobile and PC. For the description, we ask GPT-4o whether the object presented in the parsed bounding box is an app icon. If GPT-4o decides the image is an icon, it outputs one-sentence description of the icon about the potential functionality. And if not, GPT-4o will output 'this is not an icon', while still including this in the dataset. In the end, we collected 7185 icon-description pairs for finetuning.

We manually inspected the dataset to include icon-description pairs across a wide range of functions. These icons include system icons and popular software/app icons. After deduplication, we have 174 app icons in the pc platform, and 170 app icons in mobile platform. Further, we leveraged GPT-4o and conducted an analysis of the distribution of the icons. We first summarize the icons into the following types:

1. Functional Icons. These icons represent actions or functionalities users can perform. Subcategory:

• Navigation Icons: Back, forward, home, refresh, menu.

• Action Icons: Add (+), delete (trash), edit (pencil), search (magnifying glass), share, upload, download.

• System Actions: Lock, log out, power off, settings (gear icon).

2. Informational Icons. These icons convey information or statuses. Examples include: Subcategory:

• Notification Icons: Alerts, messages, updates.

• Status Icons: Battery level, network signal, Wi-Fi, Bluetooth, processing/loading (spinner).

• Error or Warning Icons: Exclamation marks, red crosses, or triangles.

3. App-Specific Icons. These icons are unique to a specific app or service.

4. Media Control Icons. These icons control media playback. Examples include play, pause, stop, fast forward, rewind, volume up, volume down.

The detail distribution of each sub-category is presented in figure 4. With this dataset, the resulting model demonstrates strong generalization on varied benchmarks and real-world applications.

We finetune BLIP-2 model for 1 epoch on the generated dataset with constant learning rate of  $ 1e^{-5} $ , no weight decay and Adam optimizer. We show a few of the qualitative examples of finetuned model vs the original model in figure 5.

### 8.2 TRAINING DETAILS OF INTERACTABLE ICON REGION DETECTION MODEL

As introduced in 3.1, we train a YOLOv8 model on the interactable icon region detection dataset. We collect in total of 66990 samples where we split 95% (63641) for training, and 5% (3349) for validation. We train for 20 epochs with batch size of 256, learning rate of  $ 1e^{-3} $ , and the Adam optimizer on 4 GPUs. We show the training curve in figure 6.

### 8.3 DETAILS OF SEEASSIGN EVALUATION

#### 8.3.1 PROMPT USED FOR GPT-4V

## GPT-4V without local semantics:

Here is a UI screenshot image with bounding boxes and corresponding labeled ID overlayed on top of it, your task is {task}. Which icon box label you should operate on? Give a brief analysis, then put your answer in the format of \n\Box with label ID: [xx]\Box\n

<div style="text-align: center;"><img src="imgs/img_in_chart_box_298_165_921_532.jpg" alt="Image" width="50%" /></div>


<div style="text-align: center;">Figure 4: Distribution of the collected icons in the icon description dataset</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td colspan="2">Before</td><td style='text-align: center;'>After</td></tr><tr><td style='text-align: center;'>an app icon with a pie chart on it</td><td style='text-align: center;'>a presentation or screen sharing application</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>the microsoft office logo is shown in a circle</td><td style='text-align: center;'>Microsoft Outlook, an email application.</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>an iphone app with an image of a flower</td><td style='text-align: center;'>Photos, a photo-sharing application.</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>an orange and white logo with a smiley face</td><td style='text-align: center;'>Discord, a messaging and voice chat application.</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>a blue app icon with a person on it</td><td style='text-align: center;'>a location or location-related function.</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>a grey and white image of a gear wheel</td><td style='text-align: center;'>Settings.</td><td style='text-align: center;'></td></tr></table>

Figure 5: Example comparisons of icon description model using BLIP-2 (Left) and its finetuned version (Right). Original BLIP-2 model tend to focus on describing shapes and colors of app icons. After finetuning on the functionality semantics dataset, the model is able to show understanding of semantics of some common app icons.

## GPT-4V with local semantics:

Here is a UI screenshot image with bounding boxes and corresponding labeled ID overlayed on top of it, and here is a list of icon/text box description: { parsed_local_semantics}. Your task is {task}. Which bounding box label you should operate on? Give a brief analysis, then put your answer in the format of \n''Box with label ID: [xx]''''n

### 8.4 Details of Mind2Web Evaluation

Here we list more details of each baseline in table 3.

SeeClick, QWen-VL SeeClick is a finetuned version of Qwen-VL on the Mind2Web training set and we report both of their numbers in their paper.



CogAgent CogAgent number is taken from the SEEAct paper (ZGK $ ^{+} $ 24), where they report cogagent-chat-hf checkpoint that is not fine-tuned on Mind2Web for experiments.

MindAct(Gen), MindAct, GPT-3.5-Turbo, GPT-4 The numbers for these baseline are taken from

<div style="text-align: center;"><img src="imgs/img_in_chart_box_298_164_923_475.jpg" alt="Image" width="51%" /></div>


<div style="text-align: center;">Figure 6: Training curves of interactable icon region detection model.</div>


the Mind2Web (DGZ $ ^{+} $ 23) paper, where they use HTML information to augment the corresponding web agent.

GPT-4V+som This model corresponds to the image annotation grounding method in SeeAct paper, where the som boxes extracted from the selected HTML elements are provided to GPT-4V to make action prediction.

GPT-4V+textual choice This corresponds to the best performing scenario in SeeAct paper (except the Oracle), that uses the selected HTML elements information in a multi-choice question format as input to the GPT-4V agent.

#### 8.4.1 QUALITATIVE EXAMPLES

We list a few more examples to demonstrate local semantics of icon function description helps GPT-4V make better action prediction in figure 7.

<div style="text-align: center;"><img src="imgs/img_in_image_box_213_171_757_790.jpg" alt="Image" width="44%" /></div>


GPT-4V without local semantics: To click on "This PC," you should operate on the icon that is labeled with the ID that corresponds to the "This PC" text in the screenshot. In this image, the "This PC" text is within the bounding box labeled with ID 1.\n\n"Box with label ID: [1]

## GPT-4V without local semantics:

To complete the task of clicking "This PC," I need to identify the bounding box that corresponds to the "This PC" icon or text. According to the provided list, "Text Box ID 0: This PC" is the description for the "This PC" element.

 $ \backslash $ n\nTherefore, the answer is:

 $ \backslash $ n\nBox with label ID: [0]

## Task: Click 'voice search icon'

GPT-4V without local semantics: To perform a voice search, you would typically look for a microphone icon, which is commonly used to represent voice input functionality. In the screenshot provided, the icon with a microphone symbol is located in the search bar area, which is labeled with the number 44.\nTherefore, the answer is:\n"Box with label ID: [44]"

GPT-4V without local semantics:

The  $ "voice search icon" $  is typically represented by a microphone icon. Looking at the list of icon/text box descriptions, the description that matches this representation is "Icon Box ID 40: a microphone icon on a white background".\n\nTherefore, the bounding box label to operate on is:\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n

<div style="text-align: center;">Figure 7: More examples of local semantics of icon functionality help with GPT-4V in grounding actions</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_215_890_387_1267.jpg" alt="Image" width="14%" /></div>


Task: click on enable button for alarm at 7:30

## GPT-4V with local semantics:

To enable the alarm set for 7:30, you should look for the toggle or switch associated with that specific alarm time. In the provided screenshot, the alarms are listed with their times, and each alarm has a corresponding switch to enable or disable it. The alarm for 7:30 is listed with the text "07:30" and has additional information indicating it is a "Daily" alarm and the time remaining until it goes off. The switch to enable this alarm is typically located on the same line as the alarm time. Based on the bounding boxes and their labels, the switch for the 7:30 alarm is likely to be the one closest to the text "07:30" and the additional information about the alarm. The bounding boxes with labels ID 23, 24, 25, and 26 appear to be switches, but we need to identify the one associated with the 7:30 alarm. Since the alarm at 7:30 is the first one listed and the switches are typically aligned with their corresponding alarms, the switch for the 7:30 alarm should be the first one in the list of switches. Therefore, the bounding box you should operate on to enable the 7:30 alarm is the one with the label ID that corresponds to the first switch. Box with label ID: [23].

<div style="text-align: center;"><img src="imgs/img_in_image_box_616_901_782_1264.jpg" alt="Image" width="13%" /></div>


## Task: More information

## GPT-4V local semantics:

Analysis: The screenshot appears to be from a music streaming application, possibly Apple Music, given the text "Apple Music Classical" in Text Box ID 3. The task instruction is to find more information about "Classical Christmas." In the screenshot, there is a section with the title "Classical Christmas" (Text Box ID 2) and a brief description (Text Box ID 7) with a "MORE" link at the end (Text Box ID 8), suggesting that clicking on this link will provide additional information about the "Classical Christmas" section.

To achieve the task, the bounding box with the description that most likely corresponds to the "MORE" link should be clicked. Based on the bounding boxes and their descriptions, Text Box ID 8 with the text "MORE" is the most relevant and accurate choice for this action.

In summary, the next action I will perform: {\\"Click ID\\":8}

<div style="text-align: center;">Figure 8: Analysis of failure cases. All the bounding boxes are labeled by which relies only on the screenshot. Left: There are in total 7 similar enable button for 7 different alarm times in the parsed screenshot. And the correct Icon ID corresponding to alarm 7:30 is 27. GPT-4V fails to make the correct prediction. Right: The ground truth region to click is the text 'MORE' inside bounding box 8. We can see that the OCR fails to detect the text 'MORE' in bold, and only detects the bounding box 8, which encompasses 'MORE'. Since the predicts the click point as the center of the box, so it the predicted click point falls outside of the ground truth region, which leads to failure in this task.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_253_584_967_918.jpg" alt="Image" width="58%" /></div>


<div style="text-align: center;">Figure 9: Analysis of failure cases. The task is to find button related to 'More information', and the ground truth is to click the three dots icon in the upper right part of the screenshot. The icon functional description model does not take into account the context of this page and interpret it as: "a loading or buffering indicator" which causes the failure.</div>