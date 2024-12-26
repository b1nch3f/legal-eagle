import os

from autogen import ConversableAgent

config_list = [
    {
        "model": "gpt-4o-mini",
        "api_type": "azure",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": "2024-08-01-preview",
    }
]

llm_config = {
    "config_list": config_list,
    "timeout": 60,
    "temperature": 0.8,
    "seed": 1234,
}

agent = ConversableAgent(
    "chatbot",
    llm_config=llm_config,
    code_execution_config=False,  # Turn off code execution, by default it is off.
    function_map=None,  # No registered functions, by default it is None.
    human_input_mode="NEVER",  # Never ask for human input.
)

data = """
'documents under Exts.1 to 13 and identified one\nmaterial object M.O .- I brash seal as against the sole\noral evidence of D.W.1 Krushna Charan Mohanty and\nsolitary documentary evidence of Ext.A by the\ndefence in support of its plea of thrusting of money\nCRLA No. 617 of 2010\nPage 4 of 20\nby the decoy in his shirt pocket. Unfortunately, the\ninformant died before he could be examined in the\ntrial.\n4.\nThe plea of the convict in the course of trial\nwas denial simplicitor and false implication with\nspecific plea that on 24.09.1997, the informant\nforcibly thrust some G.C. notes into his pocket and\nthereby, he shouted and threw away the said G.C.\nnotes, but the Vigilance people caught him and then\none Kurshna Charan Mohanty was also present there\nand was taking bath.\n5.\nAfter appreciating the evidence on record\nupon hearing the parties, the learned Special Judge,\nVigilance, Sambalpur finding the accused to have\naccepted the bribe convicted the appellant by relying\nupon the evidence on record and invoking the\npresumption U/S 20 of the Act, but the Special\nJudge, Vigilance, Sambalpur, however, acquitted the\nappellant for offences U/Ss. 13(2) r/w Section\n13(1)(d) of the Act for want of evidence with regard\nto the demand of bribe by the convict-appellant.\nCRLA No. 617 of 2010\nPage 5 of 20\n6.\nMr. S.K. Mund, learned Senior Counsel\nappearing along with Ms. A.K. Dei, learned counsel\nfor the appellant has submitted that although the\nlearned trial Court has acquitted the convict-\nappellant for offences U/Ss. 13(2) r/w Section\n13(1)(d) of the Act which is essentially the offence\nfor demand of bribe, but ignoring the failure of\nprosecution to prove the demand of bribe by the\nappellant, the Special Judge, Vigilance, Sambalpur\nerroneously convicted the appellant for offence\nromeo\nThe sprove to del\nU/S.7 of the P.C. Act, which is unsustainable in the\neye of law since the demand and acceptance are the\nprincipal ingredients of the offence and mere\nacceptance of bribe without any demand would not\nby itself establish the offence U/S. 7 of the Act. It is\nfurther submitted that when the demand is not\nestablished, invoking the presumption one under the\nSection 20 of the Act is clearly impermissible in law,\nbut the learned trial Court had erroneously invoked\nthe said presumption. Mr. Mund has accordingly,\nprayed to allow the appeal to acquit the appellant of\nCRLA No. 617 of 2010\nPage 6 of 20\nthe charge by setting aside the impugned judgment\nof conviction and order of sentence by relying upon\nthe decisions in (i) K. Santhamma v. State of\nTelangana: AIR 2022 SC 1134, (ii) Neeraj\nDutta v. State (Government of NCT of Delhi):\n2023 4SCC 731, (iii) V. Venkata Subbarao v.\nState, represented by Inspector of Police, A.P .:\n(2007) AIR (SC) 489.\nOn the other hand, Mr. M.S. Rizvi, learned\nAsc /Von the\nR\nASC (Vig.), however, while supporting the impugned\njudgment of conviction has submitted that since\nacceptance of bribe by the appellant was proved\nbeyond all reasonable doubt by unimpeachable\nevidence\nof the Sprosecut\nwitnesses, the\nconviction of the appellant for offence U/S.7 of the\nAct cannot be questioned, since the same is legally\npermissible. Accordingly, Mr. M.S. Rizvi, learned ASC\n(Vig.) has prayed to dismiss the appeal.\n7.\nAfter having bestowed a careful and anxious\nconsideration\nto the impugned judgment of\nconviction together with evidence on record keeping\nCRLA No. 617 of 2010\nPage 7 of 20\nin view the rival submissions, it admittedly appears\nthat the learned trial Court having not found any\ndirect evidence of demand of illegal gratification by\nthe appellant has acquitted him of the charge for\noffences U/Ss. 13(2) r/w Section 13(1)(d) of the\nAct, but however, it has found the appellant to have\nfailed to rebut the legal presumption as available U/S\n20 of the Act for accepting bribe of Rs.200/- as an\nillegal\ngratification from\nthe\nr accepting\ncomplainant and\nthere\nproceeded to convict the appellant for\noffence U/S 7 of the P.C. Act. In this case, the'
"""

reply = agent.generate_reply(
    messages=[
        {
            "content": f"Generate winning arguments for this data - {data}",
            "role": "user",
        }
    ]
)
print(reply)
