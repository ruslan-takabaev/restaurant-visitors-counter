import asyncio
import imageio
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.utils.exceptions import MessageNotModified, CantParseEntities
from io import BytesIO

class TelegramBotManager:
    def __init__(self, token, chat_id):
        self.bot = Bot(token=token)
        self.dp = Dispatcher(self.bot)
        self.admin_chat_id = int(chat_id)
        
        # --- Streaming attributes ---
        self.is_streaming = False
        self.frame_buffer = []
        self.MAX_BUFFER_SIZE = 50  # Number of frames for one GIF (e.g., 50 frames = 2s @ 25fps)
        self.last_sent_message_id = None
        self.gif_task = None # To hold the reference to our streaming task

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        @self.dp.message_handler(commands=['start', 'help'])
        async def send_welcome(message: types.Message):
            if message.chat.id != self.admin_chat_id: return
            await message.reply(
                "Welcome! I am the People Counter Bot.\n"
                "Commands:\n"
                "/stream - Start the live GIF stream.\n"
                "/stopstream - Stop the live GIF stream.\n"
                "/status - Get the current status (coming soon)."
            )

        @self.dp.message_handler(commands=['stream'])
        async def start_streaming(message: types.Message):
            if message.chat.id != self.admin_chat_id: return
            if self.is_streaming:
                await message.reply("A stream is already active.")
                return
            
            self.is_streaming = True
            await message.reply("Starting live stream... Please wait a few seconds for the first GIF.")
            # Start the background task for creating and sending GIFs
            self.gif_task = asyncio.create_task(self._gif_sender_loop())

        @self.dp.message_handler(commands=['stopstream'])
        async def stop_streaming(message: types.Message):
            if message.chat.id != self.admin_chat_id: return
            if not self.is_streaming:
                await message.reply("No stream is currently active.")
                return
            
            self.is_streaming = False
            if self.gif_task:
                self.gif_task.cancel() # Stop the background task
            self.frame_buffer.clear()
            self.last_sent_message_id = None
            await message.reply("Live stream stopped.")

    async def send_notification(self, text: str):
        """Sends a text notification to the admin."""
        try:
            await self.bot.send_message(self.admin_chat_id, text, parse_mode='HTML')
        except Exception as e:
            print(f"Failed to send Telegram notification: {e}")

    def add_frame_for_gif(self, frame: np.ndarray):
        """Adds a frame from the main app to our buffer for GIF creation."""
        if not self.is_streaming:
            return
        
        # Keep the buffer from growing indefinitely
        if len(self.frame_buffer) >= self.MAX_BUFFER_SIZE:
            self.frame_buffer.pop(0) # Remove the oldest frame
        
        # Convert BGR (from OpenCV) to RGB for correct colors in GIF
        rgb_frame = frame[..., ::-1]
        self.frame_buffer.append(rgb_frame)

    async def _gif_sender_loop(self):
        """The main loop that generates and sends/edits GIFs."""
        while self.is_streaming:
            # Wait until buffer is full enough to create a meaningful GIF
            if len(self.frame_buffer) < self.MAX_BUFFER_SIZE:
                await asyncio.sleep(1)
                continue

            # Create a memory file for the GIF
            gif_bytes = BytesIO()
            # Use a copy of the buffer to avoid race conditions
            frames_to_encode = list(self.frame_buffer)
            
            try:
                # Create GIF from frames in buffer. fps=25 for smooth video.
                imageio.mimsave(gif_bytes, frames_to_encode, format='gif', fps=25)
                gif_bytes.seek(0)
                media = types.InputMediaAnimation(gif_bytes)

                if self.last_sent_message_id is None:
                    # First time sending
                    sent_message = await self.bot.send_animation(self.admin_chat_id, animation=gif_bytes, caption="Live Stream")
                    self.last_sent_message_id = sent_message.message_id
                else:
                    # Edit the existing message
                    await self.bot.edit_message_media(media=media, chat_id=self.admin_chat_id, message_id=self.last_sent_message_id)

            except MessageNotModified:
                # This is okay, just means the GIF was identical.
                pass
            except CantParseEntities:
                # Happens sometimes when editing too fast, safe to ignore
                pass
            except Exception as e:
                print(f"Error in GIF sender loop: {e}")
                # Reset on error to avoid getting stuck
                self.last_sent_message_id = None
            
            # Wait a couple of seconds before sending the next update
            await asyncio.sleep(2)

    async def start_polling(self):
        """Starts the bot polling for updates."""
        print("Telegram bot started...")
        await self.dp.start_polling()

    async def stop_polling(self):
        """Stops the bot gracefully."""
        print("Stopping Telegram bot...")
        await self.dp.storage.close()
        await self.dp.storage.wait_closed()
