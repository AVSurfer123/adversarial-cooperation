
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import plotly
from plotly.graph_objs import Scatter, Figure
from plotly.graph_objs.scatter import Line
import matplotlib.pyplot as plt

def get_eval_results(file, field):
    eval_returns = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == field:
                eval_returns.append(v.simple_value)
    return np.array(eval_returns)

def graph_lines(files, figsize=(10,8), style=''):
    fields = ['Train_EnvstepsSoFar', 'Eval_AverageReturn', 'Eval_StdReturn']
    data = [[] for i in range(len(files))]

    for i in range(len(fields)):
        for j in range(len(files)):
            data[j].append(get_eval_results(files[j], fields[i]))

    data = [np.array(d) for d in data]
    
    plt.figure(figsize=figsize)
    for i in range(len(data)):
        plt.plot(data[i][0], data[i][1], style, linewidth=3.0)
        if len(data[i]) > 2:
            plt.fill_between(data[i][0], data[i][1] - data[i][2], data[i][1] + data[i][2], alpha=.3)
    return data
    

def manual_lineplot(xs, ys, title, xaxis='Iteration', yaxis='Reward', baseline=False, yrange=None):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'
    ys = np.array(ys, dtype=np.float32)
    ys_min, ys_max, ys_mean, ys_std = ys[:4]
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std
    trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
    
    data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max]

    if baseline:
        baseline_color = 'rgb(200, 0, 0)'
        trace_expert = Scatter(x=xs, y=np.repeat(ys[4], len(xs)), line=Line(color=baseline_color), name="Expert Policy")
        trace_initial = Scatter(x=xs, y=np.repeat(ys_mean[0], len(xs)), line=Line(color=baseline_color, dash='dash'), name="Behavior Cloning")
        data.extend([trace_expert, trace_initial])

    fig = Figure()
    for trace in data:
        fig.add_trace(trace)
    fig.update_layout(title=title, xaxis={'title': xaxis}, yaxis={'title': yaxis, 'range': yrange})
    fig.write_image(f'{title}_plot.png')
    return fig

# Plots min, max and mean + standard deviation bars of a population over time
def lineplot(xs, ys_population, title, xaxis='episode', yrange=None):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'
    

    if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
        ys = np.asarray(ys_population, dtype=np.float32)
        ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

        trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
        trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
        trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
        trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
        trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
        trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
        data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
    else:
        data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
    
    fig = Figure()
    for trace in data:
        fig.add_trace(trace)
    fig.update_layout(title=title, xaxis={'title': xaxis}, yaxis={'title': title, 'range': yrange})
    return fig
    # plotly.offline.plot({
    #     'data': data,
    #     'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title, 'range': yrange})
    # }, filename=os.path.join(path, title + '.html'), auto_open=False)


def write_video(frames, title, path=''):
    frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]  # VideoWrite expects H x W x C in BGR
    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
    for frame in frames:
        writer.write(frame)
        writer.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: Need tfevents file name")
    print(get_eval_results("/home/ashwin/homework_fall2020/hw1/data/q1_bc_ant_Ant-v2_09-09-2020_21-18-52/events.out.tfevents.1599711532.pabti1", 'Initial_DataCollection_AverageReturn'))
