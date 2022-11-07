# AI+X 딥러닝
### Title: DQN을 통한 mnk-game 파훼

__Members__:     
김기범, 서울 기계공학부, gbkim1997@gmail.com    
박민기, 서울 기계공학부, mgkid3310@naver.com    
오하은, 서울 컴퓨터소프트웨어학부, haeunoh.tech@gmail.com     
길준호, 에리카 전자공학부, gjh625com@naver.com 
 
__Ⅰ. Proposal (Option A)__            
* Motivation:    
2016년 3월 9일부터 15일까지 한국에서 알파고와 한국의 프로 기사인 이세돌 九 단이 바둑 대국을 진행했었다. 알파고는 알파벳의 구글 딥마인드에서 개발한 바둑 인공지능 프로그램인데, 이 대국에서 알파고는 이세돌 九 단을 4:1로 이기는 성과를 보였다. 이 일을 계기로 인공지능이 사회 보편적으로 알려지게 되었는데, 이러한 인공지능에 대한 관심에 힘입어 구체적으로 이런 인공지능 프로그램이 어떻게 작동하는지 그 원리를 이해해 보고 직접 해당 프로그램을 구현해 보고자 **DQN(Deep Q-Network)를 통한 mnk-game 파훼**라는 주제로 이번 프로젝트를 진행하기로 하였다.  
    
* What do you want to see at the end?   
mnk-game에서 k가 1, 2, 3인 경우에 확실한 파훼법이 존재하는데, k=4인 경우에는 명백한 파훼법이 존재하지 않으므로 이 경우에 대해서 DQN(Deep Q-Network)를 이용해 최선의 전략을 제공할 수 있는 모델을 완성하는 것을 이번 프로젝트의 목적으로 한다.      

__Ⅱ. Theoretical background__
 * 강화학습   
 * Q-Table (Q-Learning)
 * Neural Network
 * DQN (Deep Q-Network) 

__Ⅲ. Datasets__     
 * m, n, k = 5, 5, 4
        
__Ⅳ. Methodology__
 * input layer: m*n
 * dense 2*m*n relu
 * output m*n linear
 * tkinter

__Ⅴ. Evaluation & Analysis__      

__Ⅵ. Related Work__
 * https://ko.wikipedia.org/wiki/M,n,k-%EA%B2%8C%EC%9E%84
 * https://github.com/bruiseUneo/AI_X_DeepLearning

__Ⅶ. Conclusion: Discussion__
