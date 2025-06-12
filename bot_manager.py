import asyncio
import imageio
import numpy as np
import cv2
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from aiogram.exceptions import TelegramAPIError
from io import BytesIO

class TelegramBotManager:
    def __init__(self, token, chat_id):
        self.bot = Bot(token=token, parse_mode='HTML')
        self.dp = Dispatcher()
        self.admin_chat_id = int(chat_id)
        
        self.is_streaming = False
        self.frame_buffer = []
        self.MAX_BUFFER_SIZE = 50
        self.last_sent_message_id = None
        self.gif_task = None

        self._register_handlers()

    def _register_handlers(self):
        @self.dp.message(CommandStart())
        async def send_welcome(message: types.Message):
            if message.chat.id != self.admin_chat_id: return
            await message.reply(
                "<b>People Counter Bot (GIF Mode)</b>\n"
                "I will notify you of system events.\n\n"
                "<b>Commands:</b>\n"
                "/stream - Start the live GIF stream.\n"
                "/stopstream - Stop the live GIF stream."
            )

        @self.dp.message(Command("stream"))
        async def start_streaming(message: types.Message):
            if message.chat.id != self.admin_chat_id: return
            if self.is_streaming:
                await message.reply("A stream is already active.")
                return
            
            self.is_streaming = True
            await message.reply("Starting live stream... Please wait a few seconds for the first GIF.")
            self.gif_task = asyncio.create_task(self._gif_sender_loop())

        @self.dp.message(Command("stopstream"))
        async def stop_streaming(message: types.Message):
            if message.chat.id != self.admin_chat_id: return
            if not self.is_streaming:
                await message.reply("No stream is currently active.")
                return
            
            self.is_streaming = False
            if self.gif_task:
                self.gif_task.cancel()
            self.frame_buffer.clear()
            self.last_sent_message_id = None
            await message.reply("Live stream stopped.")

    async def send_notification(self, text: str):
        try:
            await self.bot.send_message(self.admin_chat_id, text)
        except Exception as e:
            print(f"Failed to send Telegram notification: {e}")

    def add_frame_for_gif(self, frame: np.ndarray):
        if not self.is_streaming:
            return
        
        if len(self.frame_buffer) >= self.MAX_BUFFER_SIZE:
            self.frame_buffer.pop(0)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_buffer.append(rgb_frame)

    async def _gif_sender_loop(self):
        while self.is_streaming:
            if len(self.frame_buffer) < self.MAX_BUFFER_SIZE:
                await asyncio.sleep(1)
                continue

            gif_bytes = BytesIO()
            frames_to_encode = list(self.frame_buffer)
            
            try:
                imageio.mimsave(gif_bytes, frames_to_encode, format='gif', fps=25)
                gif_bytes.seek(0)
        
                input_file = types.BufferedInputFile(gif_bytes.read(), filename="live.gif")

                if self.last_sent_message_id is None:
                    sent_message = await self.bot.send_animation(self.admin_chat_id, animation=input_file, caption="Live Stream")
                    self.last_sent_message_id = sent_message.message_id
                else:
                    media = types.InputMediaAnimation(media=input_file)
                    await self.bot.edit_message_media(media=media, chat_id=self.admin_chat_id, message_id=self.last_sent_message_id)

            except TelegramAPIError as e:
                if 'message is not modified' in str(e).lower():
                    pass
                else:
                    print(f"Telegram API Error in GIF loop: {e}. Resetting stream message.")
                    self.last_sent_message_id = None # Reset to send a new message next time
            except Exception as e:
                print(f"An unexpected error occurred in GIF sender loop: {e}")
                self.last_sent_message_id = None # Reset on other errors too
            
            await asyncio.sleep(2)

    async def start_polling(self):
        print("Telegram bot started (aiogram v3)...")
        await self.dp.start_polling(self.bot)

    async def stop_polling(self):
        print("Stopping Telegram bot...")
        pass
