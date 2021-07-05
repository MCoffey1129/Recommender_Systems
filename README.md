Recommender Systems
---

The code attached looks at three different approaches for recommending movies from a subset of the "movielens"
dataset (please note this analysis needs to be reviewed):-

   ***1. Apriori model (usually used for market based analysis):***  
      Uses support, confidence and lift (three terms which are
      outlined in the code) to determine how two products are associated with each other
   
   ***2. Collaborative filtering:***  
      Collaborative filtering leverages the power of the crowd. The intuition behind
      collaborative filtering is that if a user A likes products X and Y, and if
      another user B likes product X, there is a high chance that he will
      like the product Y as well.  

   ***3. Content based filtering:***  
      In content-based filtering, the similarity between different products is
      calculated e.g. if user 1 rates a movie of
      a specific genre really highly we would recommend a movie of the same genre
      which received a very high overall average rating
