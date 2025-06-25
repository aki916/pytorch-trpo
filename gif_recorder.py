import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
from torch.autograd import Variable
from models import *
from running_state import ZFilter

def plot_rewards(episodes, rewards, output_dir):
    """報酬の履歴をプロットして保存する関数"""
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, 'b-', alpha=0.6, linewidth=1)
    
    # 移動平均を計算して表示
    if len(rewards) > 10:
        window = min(10, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'移動平均 (window={window})')
        plt.legend()
    
    plt.xlabel('エピソード')
    plt.ylabel('平均報酬')
    plt.title('TRPO学習の進捗')
    plt.grid(True, alpha=0.3)
    
    # プロットを保存
    plot_filename = os.path.join(output_dir, 'reward_progress.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'報酬プロット保存: {plot_filename}')

def record_episode(env, policy_net, running_state, env_name, seed, select_action_func, max_steps=1000):
    """エピソードを録画してGIFとして保存する関数"""
    frames = []
    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        state, _ = reset_result
    else:
        state = reset_result
    state = running_state(state)
    
    # 環境の種類を検出
    env_type = type(env).__name__
    print(f"環境タイプ: {env_type}")
    
    def render_frame():
        """環境に応じた適切なレンダリング方法を選択"""
        try:
            # 新しいgymバージョン用
            frame = env.render(mode='rgb_array')
            if frame is None:
                raise ValueError("render() returned None")
            return frame
        except (TypeError, ValueError):
            try:
                # 古いgymバージョン用
                frame = env.render()
                if frame is None:
                    raise ValueError("render() returned None")
                return frame
            except Exception as e:
                # レンダリングできない場合はダミーフレームを作成
                print(f"Warning: 環境のレンダリングに失敗しました: {e}")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # テキストを描画
                try:
                    import cv2
                    cv2.putText(frame, f"Episode {len(frames)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, f"Env: {env_name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Step: {len(frames)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                except ImportError:
                    # cv2がない場合はnumpyでテキストを描画
                    frame[50:80, 50:200] = [255, 255, 255]  # 白い四角を描画
                return frame
    
    for step in range(max_steps):
        # 環境をレンダリングしてフレームを取得
        frame = render_frame()
        frames.append(frame)
        
        # アクションを選択
        action = select_action_func(state)
        action = action.data[0].numpy()
        step_result = env.step(action)
        
        if len(step_result) == 5:
            next_state, reward, done, truncated, _ = step_result
            episode_done = done or truncated
        else:
            next_state, reward, done, _ = step_result
            episode_done = done
            
        next_state = running_state(next_state)
        
        if episode_done:
            # 最後のフレームを追加
            final_frame = render_frame()
            frames.append(final_frame)
            break
            
        state = next_state
    
    print(f"録画完了: {len(frames)} フレーム")
    return frames

def save_episode_gif(env, policy_net, running_state, episode_num, reward_batch, 
                    output_dir, env_name, seed, select_action_func, gif_duration=5):
    """エピソードを録画してGIFとして保存する関数"""
    print(f'Recording episode {episode_num} for GIF...')
    try:
        frames = record_episode(env, policy_net, running_state, env_name, seed, select_action_func)
        
        # フレームの妥当性をチェック
        valid_frames = []
        for i, frame in enumerate(frames):
            if frame is not None and hasattr(frame, '__array_interface__'):
                valid_frames.append(frame)
            else:
                print(f"Warning: 無効なフレーム {i} をスキップ")
        
        if len(valid_frames) > 0:
            # GIFファイル名を生成
            gif_filename = os.path.join(output_dir, f'episode_{episode_num:06d}_reward_{reward_batch:.2f}.gif')
            
            # フレームレートを計算（指定された時間に合わせる）
            fps = max(1, len(valid_frames) / gif_duration)
            
            # GIFを保存
            imageio.mimsave(gif_filename, valid_frames, fps=fps)
            print(f'GIF saved: {gif_filename} ({len(valid_frames)} frames, {fps:.1f} fps)')
            return gif_filename
        else:
            print("Warning: 有効なフレームがありません")
            return None
            
    except Exception as e:
        print(f"Error saving GIF: {e}")
        print("GIF保存をスキップします")
        return None

def setup_gif_recording(output_dir):
    """GIF保存用のディレクトリを設定する関数"""
    os.makedirs(output_dir, exist_ok=True)
    print(f'GIF保存ディレクトリ: {output_dir}') 