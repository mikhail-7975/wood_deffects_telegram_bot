import cv2
import numpy as np
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.filters import Filters
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.updater import Updater
from telegram.update import Update

from wood_deffects_detector import DefectsDetector

class TelegramBot:
    def __init__(self, bot_url: str,):
        self.updater = Updater(bot_url, use_context=True)
        self.updater.dispatcher.add_handler(CommandHandler('start', self.start))
        self.updater.dispatcher.add_handler(CommandHandler('help', self.help))
        self.updater.dispatcher.add_handler(MessageHandler(Filters.photo, self.img_message_handler))
        self.updater.dispatcher.add_handler(MessageHandler(Filters.command, self.unknown))
        self.deffects_detector = DefectsDetector(detection_model_path='weights/best_640_baseline.onnx')

    def start(self, update: Update, context: CallbackContext):
        update.message.reply_text(
            "Hello sir, Welcome to the Bot.Please write"
            "help to see the commands available."
        )

    def help(self, update: Update, context: CallbackContext):
        update.message.reply_text("Введите описание мероприятия, для которого"
                                  "вам нужен мерч")

    def unknown(self, update: Update, context: CallbackContext):
        with open('results/output.png', 'rb') as file:
            update.message.reply_photo(file)
        update.message.reply_text(
            "Sorry '%s' is not a valid command" % update.message.text)

    def img_message_handler(self, update: Update, context: CallbackContext):
        # (update.message.text)
        file = update.message.photo[-1].get_file()
        # file.download()
        # image_bytes = b""
        image_bytes = file.download_as_bytearray()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        # img = cv2.imread('file_0.jpg')
        result_img = self.deffects_detector.run(img)
        cv2.imwrite('out_img.jpg', result_img)
        with open('out_img.jpg', 'rb') as file:
            update.message.reply_photo(file)

    def run(self):
        self.updater.start_polling()
