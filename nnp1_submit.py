#%%
submit_prob = net.predict(tests) # compute the softmax probs
submit_prob[0:5,:]
submit_label = np.argmax(submit_prob, axis=1)
submit_label.shape

# testing (given the ground truth)
# accuracy = np.sum(submit_label == test_y) / float(test_x.shape[0])
# print(accuracy)

#%%
# Adding index of dataset
ans = submit_label.tolist()

#%%
# Create index
submit02 = pd.DataFrame(ind)
submit02['ans'] = ans


#%%
# To csv
submit02.to_csv('./submit/submit02.csv',
              sep=' ', encoding='utf-8', index=False)

