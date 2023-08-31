from transformers import AutoTokenizer
import torch

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Filter
from aiogram.types import Message
from aiogram.utils.markdown import hbold

import warnings

warnings.filterwarnings("ignore")

TOKEN = "YOUR_TOKEN_HERE"

tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
model = torch.load("model_medium.pt", map_location=torch.device('cpu'))

dp = Dispatcher()


class MyFilter(Filter):
    def __init__(self, text: str) -> None:
        self.text = text

    async def __call__(self, message: Message) -> bool:
        return message.text != self.text


@dp.message(MyFilter("/start"))
async def message_handler(message: Message) -> None:
    user_text = message.text
    tokenized = tokenizer(user_text, return_tensors="pt")
    output_ids = model.generate(input_ids=tokenized.input_ids,
                                attention_mask=tokenized.attention_mask,
                                num_return_sequences=1,
                                max_length=1000,
                                top_k=50,
                                top_p=0.92)
    output = tokenizer.decode(output_ids[0])
    output = output.split(tokenizer.pad_token)
    await message.reply(output)


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Привет, {hbold(message.from_user.full_name)}!\n"
                         f"Я - нейросеть, обученная на диалогах из различных чатов олимпиад школьников. "
                         f"Я могу ошибаться, поэтому не переживай и попробуй немного изменить свой запрос.\n"
                         f"Чтобы начать, напиши что-нибудь")


async def main() -> None:
    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
