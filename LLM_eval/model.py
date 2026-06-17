from overlay import Qwen2_5_overlay, Flamingo_overlay, Kimi_overlay

class ASV_LLM():
    def __init__(self, model_name):
        if model_name=="Qwen/Qwen2.5-Omni-3B": self.model = Qwen2_5_overlay()  
        elif model_name=="moonshotai/Kimi-Audio-7B-Instruct": self.model = Kimi_overlay()  
        elif model_name=="nvidia/audio-flamingo-3-hf": self.model = Flamingo_overlay()
        else: exit(f"{model_name} unknown, stopping job")

        self.confidence_question = """
        These are two distinct audios.
        First think about the elements that characterize each speaker, such as their gender, accent, tone, prosody, speech rate, 
        Shortly give the characteristics for each audio
        then from those characteristics, infer the likelihood both speakers are the same. 
        Answer by Yes or No, and give a confidence score between 0 and 100:  
        0 correspond to the certainty they are from different speakers, 
        100 corresponds to the certainty that they are from the same speaker, 
        and 50 means you are uncertain.
        """
        
        self.yesno_question = """
        These are two distinct audios.
        First think about the elements that characterize each speaker, such as their gender, accent, tone, prosody, speech rate, 
        Shortly give the characteristics for each audio
        then from those characteristics, infer the likelihood both speakers are the same. 
        Answer by Yes or No.
        """

    def process(self, audio1, audio2, return_logits=False, confidence_question=True):
        if confidence_question: return self.model.process(audio1, audio2, question=self.confidence_question, return_logits=return_logits)
        else: return self.model.process(audio1, audio2, question=self.yesno_question, return_logits=return_logits)
        



