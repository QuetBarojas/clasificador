{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "512d4d95",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, render_template\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "\n",
    "@app.route(\"/\")\n",
    "def index():\n",
    "    return render_template('index1.html')\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(host='0.0.0.0', port=8000, debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ec157e2f",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (Temp/ipykernel_3664/1226559031.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"C:\\Users\\QUET\\AppData\\Local\\Temp/ipykernel_3664/1226559031.py\"\u001b[1;36m, line \u001b[1;32m1\u001b[0m\n\u001b[1;33m    export FLASK_APP=Untitled\u001b[0m\n\u001b[1;37m           ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a03c78b0",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (Temp/ipykernel_3664/4033298694.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"C:\\Users\\QUET\\AppData\\Local\\Temp/ipykernel_3664/4033298694.py\"\u001b[1;36m, line \u001b[1;32m1\u001b[0m\n\u001b[1;33m    export FLASK_ENV=development\u001b[0m\n\u001b[1;37m           ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b9d358c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
