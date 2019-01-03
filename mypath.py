class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/path/to/Segmentation/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif database == 'sbd':
            return '/path/to/Segmentation/benchmark_RELEASE/' # folder that contains dataset/.
        elif database == 'cityscapes':
            #return '/data/teddy/Datasets/kitti/'
            #return '/data/teddy/Datasets/cityscapes/'  # folder that contains leftImg8bit/
            return '/data/teddy/Datasets/cityscapes_plus_kitti/'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
