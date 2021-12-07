
def return_organisation_links(new_order):
    """
    This can be updated to include the most relevant organisation links
    for the required country or region.
    Note the order of this list needs to vary with the F1-score so may be
    best implemented as a dictionary?
    """

    links = ["http://"+i+".com" for i in new_order]

    return links