

social_relationships_lookup = {'unknown' : 0.0, 'adversary' : 0.0, 'acquaintance' : 0.25,
                                'strong acquaintance' : 0.5, 'friend' : 0.5,
                                 'close friend' : 0.75, 'partner' : 1.0,
                                  'family' : 1.0}


def set_social_relationships_weights(social_context):
    """
    Converts the social relationship description text (the field relationship_edited
    in the passed dataframe into their corresponding 
    weights according to the lookup table defined in this file.
    Note: the conversion happens in place (no return value) and the passed 
    dataframe is changed
    
    Args:
        social_context(pandas.DataFrame): (from, to, social_capital, tie_strength,
         social_similarity, social_context_similarity, sympathy, domain_expertise,
          social_hierarchy, relationship, relationship_edited)
            
    """
    social_relationships_edited = social_context['relationship_edited']
    for i in social_relationships_edited.index:
        social_relationship = social_relationships_edited[i]
        social_relationships_edited.at[i] = social_relationships_lookup[social_relationship]
    social_context['relationship_edited'] = social_context['relationship_edited'].astype(float)
    


def remove_social_relationship_field(social_context):
    """
    Removes the field 'relationship' from the social_context DataFrame.
    Returns the social_context DataFrame without the 'relationship' column
        
    Args:
        social_context(pandas.DataFrame): (from, to, social_capital, tie_strength,
         social_similarity, social_context_similarity, sympathy, domain_expertise,
          social_hierarchy, relationship, relationship_edited)
    """
    return social_context.drop('relationship', axis=1)
