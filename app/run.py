from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from summary.summary import summary
from classification.classification import classification
import joblib
import pandas as pd

def main(passage):
    MODEL_NAME = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)
    result_sum = summary(tokenizer, model, passage)

    data = pd.read_csv('classification/vector_data.csv',encoding = "ISO-8859-1")
    clf = joblib.load('classification/best.pkl')
    result_clf = classification(data, clf, passage)
    
    print(result_sum)
    print(result_clf)
    return result_sum, result_clf
    


if __name__ == "__main__":
    passage = """He trudged out of the main entrance of the rental apartment with  his son. I could see the magnolia tree next to the main gate clearly showing itself in the streetlight.   Magnolia flowers sometimes wither quickly . Before I knew it, the petals lost their white color and were drooping like wet towels and dried branches. He was walking  toward the  town bus station when he suddenly realized that he had not come out with his wallet. There was only one credit card and one receipt in the pocket. His credit card didn't have a separate transportation card function. Because it was a credit card issued a long time ago. He was about to go back home again when he saw the back of his son's head walking with his head down. Let's just walk. He decided so in his mind.  I think it'll take about 30 minutes if I walk. He thought for no reason that it was a way to further his son's heart. He passed by the  town bus station without saying a word. The son walked with him without asking a word. The fourth-grade elementary school son had a large Lego box in both hands. It was past ten o'clock at night.
He had  a minor accident last night. Working at a  logistics center on the  outskirts of the  city, he  had a  company dinner with  his co-workers after a  long time, waited for a town bus with a  slightly unquiet face  after an early dinner, and suddenly, almost impulsively, walked into a large mart just across the street. A week ago, I went grocery shopping with my wife and son, and I remembered my son standing in front of the Lego corner for a long time. Why? Do you want this? He asked his  son with a playful tap on the shoulder. Then, almost at the same time, I looked at the price tag of the Lego box that my son was looking at. 299,000 won. He was a little embarrassed, but he purposely tightened his legs so that he wouldn't be caught embarrassed. The son looked at his face and laughed. It's  like a month in my house. The son then hit his waist with his  shoulder.
To be  honest, I thought it had passed by like  that. So it was right that he himself did not understand his behavior properly last night. He was a little upset when he bought the Lego box with a credit card. The words "repeat, repeat" kept coming to my mind. Spring is  coming back, spring is  coming back, he'll  take the bus to  and from work like  his father did, he'll always work hard to pay rent and living expenses, and he'll think about a lot of things every time he buys a spring jumper, and then he'll feel sick in vain...I think he kept thinking that. He was even more angry with himself because he didn't trust himself to repeat such thoughts while buying his son a Lego.
He talked to  his son  while waiting for the  intersection light. Junghoo, should we not change this? His son looked at the traffic light across the street and picked up the horse. What if I don't change it? You don't want to go home? It was  only this evening that  his wife noticed the identity of the Lego box. His wife, who has been working as a study paper teacher for two years, often couldn't have dinner together. When I woke up in the middle of the night, my wife, who couldn't take off her stockings, would fall asleep as if she had fainted on the edge of the bed. Shall we go to a dry sauna? We can sleep there. His wife gave him and his son a Lego box and a receipt and asked him to get a refund right away. Are you out of your mind or not? You're a kid? My wife said so. Dad can go to work right away from the jjimjilbang, but I have to go home again and take my school bag. The son spoke in a feeble voice.
They crossed the  crosswalk and started walking . Junghoo, you hate your dad, right? He asked his son. I know. Why did you buy this? When did I ask you to buy me... He walked at the pace of his son's steps.  I just wanted  to buy it for you, well...His son did not say. But it's not cold even if you walk because it's spring, right? The moment he said so, his son began to sob. Drops of tears fell on the Lego box. The son continued to steal the teardrops that fell on the Lego box, but did not stop crying. He was at a loss because he was embarrassed, but on the other hand, somehow the scenery itself was familiar, so he just stood still. There was a spring night when he cried like that, too.
    """

    
    main(passage)
    
    