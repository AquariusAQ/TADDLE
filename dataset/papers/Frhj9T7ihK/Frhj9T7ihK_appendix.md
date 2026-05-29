## A APPENDIX

You may include other additional sections here.

Authors may use as many pages of appendices (after the bibliography) as they wish, but reviewers are not required to read the appendix.

## B META-PARAMETER SETTINGS

We performed a standard 10-fold cross-validation on the dataset, training the model until test set accuracy plateaued for a fixed number of epochs (40). Due to the small dataset size and the fact that the goal of this experiment is not to precisely gauge the model's prediction accuracy, we did not allocate a separate validation set. The model was trained in batches of 4 using the Adam optimizer with a learning rate of 0.01. The top-k case selection layer was disabled, as enabling it led to lower accuracy. Although some parameter choices (e.g., the relatively high learning rate) may seem unconventional in typical machine learning setups, they are justified and discussed in Ye et al. (2024). Most importantly, additional configurations were employed to enhance the model's interpretability for practitioners:

• All cases share the same feature weights  $ (w_{x\delta_{i}} = w_{y\delta_{i}} = w_{\delta_{i}} $  for any two cases x and y) to reduce the number of parameters. This approach applies global feature weighting, where each feature is assigned a single weight across all cases.

• For each case x, we initialize the feature weights as  $ w_{x\delta_{i}} = 1 $ , the case bias as  $ b_{x} = 50 $ , the case weight as  $ w_{(x,L)} = 1 $ , and all class biases as  $ b_{L} = 1 $ .

## C DATASET DESCRIPTION

The dataset is a collection of answers from 157 undergraduate students to a survey of true/false questions. The questions are designed according to the Diagnostic and Statistical Manual of Mental Disorders by a psychologist (Orozco-del Castillo et al., 2021). The questions are listed below.

1. Most of the time I have difficulty concentrating on simple tasks

2. I don’t feel like doing my daily duties

3. My friends or family have told me that I look different

4. When I think about the future it is difficult for me to imagine it clearly

5. People around me often ask me how I feel

6. I consider that my life is full of good things

7. My hobbies are still important to me

8. I’m still as punctual as I have always been

9. If I had the chance, I would spend all day in my bed

8. I'm still as punctual as I have always been

9. If I had the chance, I would spend all day in my bed

10. I have found that I can spend a lot of time scrolling the screen of my cell phone without searching or stopping at anything in particular

11. When someone asks me something, I have noticed that I take longer than normal to respond

12. I have noticed my body shaken without any cause

13. I felt more encouraged to do my daily activities before

14. Sometimes I wake up sad and I can't explain why

15. In recent months I usually reproach myself for things from the past

16. I think my thoughts are strange or different from before

17. I feel guilty about the decisions that I have made

18. I don't feel as comfortable with my body as I did before

19. I don't feel successful compared to others

20. It is difficult for me to make decisions even if they are simple

21. I'm capable of achieving what I propose to myself

22. It is not difficult for me to understand something the first time

23. I have thought more than before about what my death would be like

24. Being dead seems to be a solution to some problems

25. I would rather stay home than go out with my friends

26. I like to attend family gatherings

27. I feel excited when thinking about my life project

28. The decisions I have made so far have been the right ones

29. I am able to carry out my activities as I have always been

30. I like to be in touch with my friends and family through social media

31. It is easy for me to choose a photograph of myself to show it on social media

32. I am proud of what I have achieved so far

33. I have trouble remembering things easily

34. In recent months I have had discussions with my schoolmates or colleagues

35. I constantly imagine that something will go wrong at my work or at school

36. I am afraid of being wrong when doing my homework

37. I'm not too worried about what might happen in a few weeks

38. Lately it's hard for me to calm down

39. Everything will be alright

40. I can easily blank my mind

41. I am bothered by insignificant things that were not important before

42. I find it uncomfortable to be in a crowded place

43. Sometimes I feel trapped

44. I am easily frightened by unexpected noises

45. I have difficulties to do one task at a time

46. I have the feeling that I am forgetting to do something

47. I can clearly express to others how I feel

48. I can sleep easily

49. I enjoy every moment of the day

50. I imagine that at any moment a disaster of nature may occur

51. Sometimes I feel like I get tired easily

52. Being locked in an elevator would be the worst thing that could happen to me

53. I'm bothered by people walking slowly in front of me

54. I don't usually get upset if something doesn't go as expected

55. Sometimes it is as if some conversations with friends or family become interrogations

56. I manage my schedule as I always have

57. It bothers me to feel that people on the street approach me

58. I have no difficulty understanding what people explain to me

59. I consider that I am good at controlling my emotions

60. In new situations I feel calm and encouraged

61. Sometimes I forget what I wanted to say because I have several thoughts at the same time

62. I would like to know what will happen in the future

63. When I get angry I can easily explode

64. I can put down my cell phone and dedicate myself to reading without distractions

65. I worry that people will not understand what I mean

66. Sometimes I do not listen to what people say to me because I am thinking about other things

67. I get angry easily

68. I'm afraid that something bad could happen to me

69. It is not important for me to meet set dates

70. I like to think clearly before giving my opinion

71. I use lies just to get out of certain problems

72. If I have the opportunity to get in line to avoid wasting time, I do it

73. I have difficulty making elaborate plans

74. People have problems because of themselves

75. No more people are important to me than others

76. Laws are not as important as others think

77. I would regret betraying a friend

78. I prefer that a negotiation supports the largest possible number of people involved

79. It is easy for me to work in a team

80. It is important to help people when they need it

81. I have punched someone or thought of doing it

82. If it was necessary I would pretend to be someone else to get something

83. I consider it important to ensure my physical safety and that of those around me

84. After an argument I usually go over what happened in my head

85. I have a hard time controlling myself when I get angry

86. Loyalty is important

87. If I can help a person I will stop what I'm doing to help them

88. Sometimes people need physical force to understand

89. It makes me laugh when my superiors at school or at work demand something

90. Deceiving people is not wrong if it is to achieve something important

91. I like to greet my neighbors

92. It does not seem serious to me to have some debts

93. People steal because they have needs

94. I lose control easily

95. Neighbors must put up with each other's noises without complaining

96. Littering on public roads is wrong

97. People who commit crimes have their reasons for doing it

98. It is normal to change jobs several times a year

99. It is important to respect turns

100. I could pretend to be someone else to achieve what I want

101. I consider it important that all people have the same rights

102. I have a hard time taking "no" for an answer

## D DATA PREPROCESSING

The dataset was used to train neural networks for depression screening and then later used for an explainable AI challenge in the Explainable AI Challenge at the 2022 International Conference on Case-Based Reasoning (Wilkerson et al., 2022). We obtained a version of the dataset with 104 cases with 102 attributes and a risk score (from 1 to 5) calculated from the number of physical symptoms related to depression. We oversampled less frequent classes to counteract class imbalance and transformed risk scores into three classes (low, medium, and high risk of depression) following the example of Wilkerson et al. (2022). Our final dataset contains 117 cases; Each case has 102 features of binary values and a class label of 0, 1 or 2, representing low, medium and high risk.

## E INTERVIEW QUESTIONS

Following are the questions we used when interviewing the 10 participating practitioners.

### E.1 DEMOGRAPHIC QUESTIONNAIRE

1. What's your full name?

2. What's your gender?

3. What's your age?

4. What's your race?

5. What's your sexual orientation?

6. What's your highest degree and for how long have you been licensed?

7. What's your professional field (Counseling, clinical, social work, school psychology, etc.)?

8. What's your current job?

9. Which state are you currently living in?

10. What are your primary clinical populations?

11. What are your primary theoretical orientations?

12. Please rate your knowledge about AI technology from 0-10 (0 being absolutely no knowledge, 10 being complete expertise).

### E.2 QUALTRICS QUALITATIVE QUESTIONS

## As the preliminary user:

1. Regarding the tunable experience you just had, did you feel like the model has become more useful clinically after you tuned the feature weights? Please elaborate.

2. Did you feel like the model has become more clinically trustworthy? Please elaborate.

3. Please provide any other comments or thoughts about your experience trying out our algorithm just now.

## As a practitioner thinking about using this model for clinical diagnosis:

1. If you were to use this AI algorithm, would the tunable feature be useful for you?

2. What would be the strengths and concerns you have about using this tunable feature?

3. Since our model can detect bias, if the model’s explanations differ from your clinical judgment, what would you do?

4. Since our model is fully interpretable and tunable, would this algorithm be more ethically trustworthy?

5. What other factors should we consider in improving this model for clinical diagnosis?