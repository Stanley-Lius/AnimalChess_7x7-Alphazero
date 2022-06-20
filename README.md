# AnimalChess_7x7-Alphazero
***使用7x7而不是3x4棋盤大小的動物將棋，其他規則都跟[3x4](https://github.com/Stanley-Lius/AnimalChess_3x4-Alphazero)一樣***  
***會改變的有game.py中棋盤的大小、敵我方棋勢、可以下的棋步選擇，以及對偶網路dual_network.py中的輸入輸出，還有就是在遊玩GUI畫面中的棋盤大小以及棋子所在位置會不同(這專案中的7multiply7.py就是human_play.py，只是棋盤配置及棋子所在位置不同)***  
# AnimalChess_7x7-Alphazero
使用alphazero架構訓練AI遊玩動物將棋
# 大二下python期末專題
## 組員:  
蔡旺霖  
## 專題名稱:  
以AlphaZero實作賽局遊戲AI
## 參考書籍  
[這裡](https://www.tenlong.com.tw/products/9789863126515)
# AlphaZero主要理念  
AlphaZero主要是由深度學習、加強式學習以及賽局數演算法組合而成  
## 深度學習  
用來預測最佳棋步的直覺，簡單來說就是以***訓練集***(以前的資料)來預測下一步該如何走  
## 強化式學習  
藉由自我對弈以及蒙地卡羅搜尋樹來獲得回饋值，使代理人能藉由回饋值計算出沒一步棋路的***價值***，從而做出最好的選擇  
## 賽局樹演算法  
預知能力，相較於深度演算法是以當前的棋勢來去局部推演出接下來所可能走的棋步(預測所有可能的未來)，然後把其中最有可能導致獲勝的棋步選擇出來  
# 程式碼大略介紹  
## game.py
定義動物將棋遊戲規則，棋盤大小為3*4，有[獅子](https://github.com/Stanley-Lius/AnimalChess_3x4-Alphazero/blob/main/piece4.png)、[小雞](https://github.com/Stanley-Lius/AnimalChess_3x4-Alphazero/blob/main/piece1.png)、[鹿](https://github.com/Stanley-Lius/AnimalChess_3x4-Alphazero/blob/main/piece3.png)和[象](https://github.com/Stanley-Lius/AnimalChess_3x4-Alphazero/blob/main/piece2.png)四種動物所組成  
小雞只能往前走；鹿可以往前後左右四個方向走；象則是可以往對角線方向走  
獅子則是總和以上三種，八個方位都可以走，如同超連結中途圖片紅點所示  
遊戲中有持駒規則，亦即如果我們吃掉了對方的棋子，在下個我方回合可以把它放到棋盤任意空格子，且會變為我方棋子  
遊戲***勝利條件***是將對手的獅子吃掉，或者下超過三百回合即算平手  
***此程式***主要是用來回傳現在的局勢以及可以下的棋子、可以下的點，還有判斷獅子是否被吃掉的終局條件  
## dual_network.py  
定義了對偶網路的架構，由一個捲基層、16個殘差塊以及一個池化層所組成；輸出有兩個，分別是***策略輸出***(策略，這裡是每個棋盤格子可以被下的機率所組合而成的陣列，陣列合為1)，  以及***價值輸出***(局勢價值，代表了用來預測當前盤面所會導致的勝負)，  
訓練集的部分是由接下來會講道的self_play.py所產生  
## pv_mcts.py  
定義蒙地卡羅樹搜尋法的實作結構，由於此演算法很像binary tree，所以我們會定義一個資料結構node用來儲存目前局勢、試驗次數、價值以及子節點。
此程式最終目的是取的目前節點的子節點中，找出***最大試驗次數***的子節點(會使用波茲曼函數以及PUCT公式來讓棋步增加一點不確定性，也就是增加棋路的選擇)  
***若沒有子節點的話則是由深度學習網路所訓練出來的模組進行預測***    
## self_play.py  
定義了自我對奕模組，由目前最好的深度學習模組(一開始的最好模組當然都是***隨機參數***，會經由訓練慢慢變強)以及pv_mcts.py的預知能力結合而成，由電腦
v.s. 電腦來產生訓練集，預設是自我對奕500次  
## train_network.py  
以self_play.py自我對奕所儲存的訓練集訓練對偶網路  
## evaluate_network.py  
定義了評估測試的環境，由目前最新訓練出來的模組與之前最好的模組進行對奕，如過***最新模組勝率***大於50%，則將最佳模組替換成目前最新的模組  
## train_cycle.py
定義了訓練的次數並執行，跑一次self_play.py、train_network.py、evaluate_network.py為一整個完整的流程，此程式會執行這整個流程10次，以獲得較完整的學習效果。  ***model資料夾中已訓練好的模組為跑過三次train_crycle.py所得出的模組，也就是說會跑上面提到的流程30次***  
## 7multiply7.py  
定義電腦GUI遊玩介面，以twinter繪製棋盤以及棋子，以便測試以及遊玩。  
# 建議  
***如果要跑訓練的話可以在Google Colab上執行，可以使用免費的GPU資源；但是若要執行human_play.py遊玩則一定要用使用本地端電腦，因為Colab不支援在雲端開新分頁***



