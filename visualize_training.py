#!/usr/bin/env python3
"""
TRPO学習の進捗をリアルタイムで可視化するスクリプト
"""

import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np

def load_reward_data(output_dir):
    """報酬データを読み込む関数"""
    episodes = []
    rewards = []
    
    # main.pyで保存されたデータを読み込む
    # 実際の実装では、main.pyからデータをリアルタイムで受け取る必要があります
    # ここでは例として、ファイルから読み込む方法を示します
    
    return episodes, rewards

def create_realtime_plot():
    """リアルタイムプロットを作成"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('エピソード')
    ax.set_ylabel('平均報酬')
    ax.set_title('TRPO学習の進捗 (リアルタイム)')
    ax.grid(True, alpha=0.3)
    
    line, = ax.plot([], [], 'b-', alpha=0.6, linewidth=1, label='報酬')
    moving_avg_line, = ax.plot([], [], 'r-', linewidth=2, label='移動平均')
    
    ax.legend()
    
    def init():
        line.set_data([], [])
        moving_avg_line.set_data([], [])
        return line, moving_avg_line
    
    def animate(frame):
        # ここで実際のデータを読み込む
        episodes, rewards = load_reward_data('gifs')
        
        if len(rewards) > 0:
            line.set_data(episodes, rewards)
            
            # 移動平均を計算
            if len(rewards) > 10:
                window = min(10, len(rewards) // 10)
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                moving_avg_line.set_data(episodes[window-1:], moving_avg)
            
            # 軸の範囲を自動調整
            ax.relim()
            ax.autoscale_view()
        
        return line, moving_avg_line
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=None, 
                        interval=5000, blit=True)  # 5秒ごとに更新
    
    plt.show()
    return anim

def monitor_gif_directory(output_dir='gifs'):
    """GIFディレクトリを監視して最新のGIFを表示"""
    print(f"GIFディレクトリ '{output_dir}' を監視中...")
    print("Ctrl+C で終了")
    
    last_files = set()
    
    try:
        while True:
            if os.path.exists(output_dir):
                current_files = set(os.listdir(output_dir))
                new_files = current_files - last_files
                
                for file in new_files:
                    if file.endswith('.gif'):
                        filepath = os.path.join(output_dir, file)
                        print(f"新しいGIFファイル: {file}")
                        print(f"  パス: {filepath}")
                        print(f"  サイズ: {os.path.getsize(filepath)} bytes")
                        print(f"  作成時刻: {time.ctime(os.path.getctime(filepath))}")
                        print("-" * 50)
                
                last_files = current_files
            
            time.sleep(2)  # 2秒ごとにチェック
            
    except KeyboardInterrupt:
        print("\n監視を終了しました")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TRPO学習の可視化ツール')
    parser.add_argument('--mode', choices=['realtime', 'monitor'], default='monitor',
                        help='可視化モード (default: monitor)')
    parser.add_argument('--output-dir', type=str, default='gifs',
                        help='出力ディレクトリ (default: gifs)')
    
    args = parser.parse_args()
    
    if args.mode == 'realtime':
        print("リアルタイムプロットを開始...")
        create_realtime_plot()
    elif args.mode == 'monitor':
        monitor_gif_directory(args.output_dir) 