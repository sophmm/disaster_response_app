import pandas as pd
import plotly.graph_objs as go


def per_of_message_per_cat(df):
    return (df.sum(axis=0) /df.sum(axis=0).sum()) * 100

def return_figures(df, eval_df):
    """
    Creates 2 plotly visualizations

    Returns:
        list (dict): list containing the 2 plotly visualizations
    """

    graph_one = []
    eval_df2 = eval_df.sort_values(by='f1_score', ascending=True)
    eval_df2.reset_index(inplace=True)
    fscores = eval_df2['f1_score'] * 100
    cat_fscore = eval_df2['index']

    all_colours = ['mediumseagreen' if i > 60 else 'orange'
    if (i >= 30) and (i <= 60) else 'indianred' for i in fscores]

    graph_one.append(
        go.Bar(
            y=fscores,
            x=cat_fscore,
            marker=dict(
                color=all_colours,
                line=dict(color='darkslategray', width=0.5))
        )
    )

    layout_one = dict(title='Classification confidence (%)',
                      yaxis=dict(title='%'),
                      )


    graph_two = []
    perc_mess = per_of_message_per_cat(df.drop(columns=['message', 'genre','shops','tools']))
    perc_df = pd.DataFrame(perc_mess, columns=['perc_mess']).sort_values(by='perc_mess', ascending=True)

    perc_df.reset_index(inplace=True)

    perc_mess_to_plot = perc_df['perc_mess']
    cat_perc_mess = perc_df['index']

    graph_two.append(
        go.Bar(
            x=cat_perc_mess,
            y=perc_mess_to_plot,
            marker=dict(
                color='navy',
                line=dict(color='darkslategray', width=0.5))
        )
    )

    layout_two = dict(title='% of messages used for training the model',
                      yaxis=dict(title='%'),
                      )

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))

    return figures
