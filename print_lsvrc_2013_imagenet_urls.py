"""Print out list of imagenet urls."""

import bs4
import logging
import urllib2

CHALLENGE_2013_URL = 'http://www.image-net.org/challenges/LSVRC/2013/browse-synsets'
IMAGENET_API_URL = 'http://imagenet.stanford.edu/api/text/imagenet.synset.geturls'

def get_classes(page_url):
    classes = []
    try:
        webpage = urllib2.urlopen(page_url)
        soup = bs4.BeautifulSoup(webpage.read().decode('utf8'))
    except:
        logging.error("Error getting %s" % page_url)
        raise
    for anchor in soup.findAll('a'):
        img_url = anchor.get('href')
        if 'wnid' in img_url:
            classes.append(img_url.split('=')[1])
    return classes

def get_img_urls(img_list_url):
    webpage = urllib2.urlopen(img_list_url)
    return [line.strip('\n\r') for line in webpage if line.startswith('http')]

if __name__ == '__main__':
    for id in get_classes(CHALLENGE_2013_URL):
        print '\n'.join(get_img_urls('%s?wnid=%s' %(IMAGENET_API_URL, id)))
