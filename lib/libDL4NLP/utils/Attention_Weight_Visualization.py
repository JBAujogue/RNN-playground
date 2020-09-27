

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def AttentionViewer(attention, row_labels, col_labels) : 
    '''attention  = np.array of size (num_rows, num_columns)
       row_labels = list of (str)
       col_labels = list of (str)
    '''
    def heatmap(data, row_labels, col_labels, ax = None, cbar_kw = {}, cbarlabel = "", **kwargs):
        if not ax: ax = plt.gca()
        # Plot the heatmap
        im = ax.imshow(data, **kwargs)
        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")
        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        return im

    def annotate_heatmap(im, data = None, valfmt = "{x:.2f}", textcolors = ["black", "white"], threshold = None, **textkw):
        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()
        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.
        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)
        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
        return texts
    
    # -- main --
    fig, ax = plt.subplots()
    im      = heatmap(attention, row_labels, col_labels, ax=ax, cmap="YlGn", cbarlabel="harvest [t/year]")
    texts   = annotate_heatmap(im, valfmt="{x:.2f}")
    fig.tight_layout()
    plt.show()
    return


def AttentionViewerOnWords(attention, row_labels, col_labels, 
                           colors = 'Reds', n = 8) : 
    '''attention  = np.array of size (num_rows, num_columns)
       row_labels = list of (str)
       col_labels = list of (str)
    '''
    def generateColors(colors, n):
        colors = plt.get_cmap(colors)
        Triplets = []
        for i in range(n) :
            triplet = [int(j * 256) for j in colors(i/10)[:3]]
            Triplets.append(triplet)
        return Triplets
    
    def weight2color(weight, triplets):
        n = len(triplets)
        for i in range(n):
            if weight >= i/n and weight <= (i+1)/n : 
                return triplets[i]
            
    def addColor(texte, RGB = (100,100,100)):
        new_texte = '\x1b[48;2;'  + str(RGB[0]) + ";" + str(RGB[1]) + ";" + str(RGB[2]) + "m"  + texte + "\x1b[0m"
        return new_texte
    
    # -- main --
    Triplets = generateColors(colors, n)
    Colored_text = ''
    sep = ' - ' if attention.shape[0] > 1 else ''
    for i in range(attention.shape[0]) :
        #Colored_text += row_labels[i] + sep
        for j in range(attention.shape[1]) :
            color = weight2color(attention[i, j], Triplets)
            Colored_text += addColor(col_labels[j], color) + ' '
        Colored_text += sep + row_labels[i] + '\n' 
    print(Colored_text)
    return


def HANViewerOnWords(attn_words, attn_sentences, sentences, 
                                       colors = 'Reds', n = 8) : 
    '''attn_words = [1D np.array]
       attn_sentences = 1D np.array
       sentences = [[str]]
    '''
    def generateColors(colors, n):
        colors = plt.get_cmap(colors)
        Triplets = []
        for i in range(n) :
            triplet = [int(j * 256) for j in colors(i/10)[:3]]
            Triplets.append(triplet)
        return Triplets
    
    def weight2color(weight, triplets):
        n = len(triplets)
        for i in range(n):
            if weight >= i/n and weight <= (i+1)/n : 
                return triplets[i]
            
    def addColor(texte, RGB = (100,100,100)):
        new_texte = '\x1b[48;2;'  + str(RGB[0]) + ";" + str(RGB[1]) + ";" + str(RGB[2]) + "m"  + texte + "\x1b[0m"
        return new_texte
    
    # -- main --
    Triplets = generateColors(colors, n)
    Colored_text = ''
    for i, s in enumerate(sentences) :
        s_color = weight2color(attn_sentences[i], Triplets)
        Colored_text += addColor('  ', s_color) + ' '
        for j, w in enumerate(s) :
            color = weight2color(attn_words[i][j], Triplets)
            Colored_text += addColor(w, color) + ' '
        Colored_text += '\n' 
    print(Colored_text)
    return
