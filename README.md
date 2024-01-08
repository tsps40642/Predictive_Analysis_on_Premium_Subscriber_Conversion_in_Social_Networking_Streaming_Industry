# Predictive Analysis on Premium Subscriber Conversion in Social Networking Streaming Industry

## Overview
Website XYZ, a music-listening social networking website, follows the “freemium” business model, offering basic services for free and providing a number of additional premium capabilities for a monthly subscription fee.  

We are interested in predicting which people would be likely to convert from free users to premium subscribers in the next 6 month period, if they are targeted by the promotional campaign. The specific task is to build the best predictive model for predicting likely adopters (that is, which current non-subscribers are likely to respond to the marketing campaign and sign up for the premium service within 6 months after the campaign). 

## Dataset
A labeled dataset containing 41,540 records (1540 adopters and 40,000 non-adopters) from the previous marketing campaign is provided. 
Each record represents a different user who was targeted in the previous marketing campaign, described with 25 attributes. 

The following is a brief description of the attributes (attribute name/type/explanation):  
• adopter / binary (0 or 1) / whether a user became a subscriber within the 6 month period after the marketing campaign (target variable)   
• user id / integer / unique user id  
• age / integer / age in years  
• male / integer (0 or 1) / 1 – male, 0 – female  
• friend cnt / integer / numbers of friends that the current user has  
• avg friend age / real / average age of friends (in years)  
• avg friend male / real (between 0 and 1) / percentage of males among friends  
• friend country cnt / integer / number of different countries among friends of the current user  
• subscriber friend cnt / integer / number of friends who are subscribers of the premium service  
• songsListened / integer / total number of tracks this user listened (or reported as listened)  
• lovedTracks / integer / total number of different songs that the user “liked”  
• posts / integer / number of forum or discussion board posts made by the user  
• playlists / integer / number of playlists created by the user  
• shouts / integer / number of wall posts received by the user  
• good country / integer (0 or 1) / country type of the user: 0 – countries where free usage is more limited, 1 – less limited  
• tenure / integer / number of months since the user has registered on the website  

There're also a number of attributes with the name delta <attr-name> where <attr-name> is one of the attributes mentioned in the above list.  
Such attributes refer not to the overall number but the change to the corresponding number over the 3-month period before the campaign.  
For example, consider attribute delta friend cnt. If, for some user, friend cnt = 50, and delta friend cnt = –5, it means that the user had 50 friends at the time of the previous marketing campaign, but this number reduced by 5 during the 3 months before the campaign (i.e., user had 55 friends 3 months ago).  

## Steps
1. Based on the description of the business, select a proper performance metric for model evaluation and justify your selection from a business perspective.
2. Build the best model that achieves highest performance on the metric of your selection.
3. Note that the class distribution is highly imbalanced (with class 1 being the minority class), you are highly encouraged to consider sampling techniques (e.g., oversampling the minority class).
4. Also present the model to a management-oriented audience.
