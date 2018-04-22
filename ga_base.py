import math
import pickle
import random
import string
import webbrowser
from bisect import bisect
from pprint import pformat

import numpy as np
from PIL import Image, ImageDraw


def compare_images(im1, im2):
    im1_rgb = np.array(get_thumbnail(im1.convert('RGB')).getdata())
    im2_rgb = np.array(get_thumbnail(im2.convert('RGB')).getdata())
    difference = np.linalg.norm(im1_rgb - im2_rgb)
    return difference


def rand_instructs(size, numpolys, numcorners=3):
    xy_instructs = np.random.random((numpolys, numcorners * 2))
    xy_instructs[:, np.array(range(0, numcorners * 2, 2))] *= size[0]
    xy_instructs[:, np.array(range(0, numcorners * 2, 2)) + 1] *= size[1]
    color_instructs = np.random.random((numpolys, 4)) * 256
    instructs = np.concatenate((xy_instructs, color_instructs), axis=1)
    return instructs.astype(int)


def get_thumbnail(image, size=(128, 128), stretch_to_fit=True, greyscale=False):
    " get a smaller version of the image - makes comparison much faster/easier"
    if not stretch_to_fit:
        image.thumbnail(size, Image.ANTIALIAS)
    else:
        image = image.resize(size)  # for faster computation
    if greyscale:
        image = image.convert("L")  # Convert it to grayscale.
    return image


def growPolygon(poly, imsize, minsize):
    f = 1.1
    while polygonArea(poly[:6]) < minsize:
        # f += 0.01
        # print(f)
        poly = (poly * np.array([f, f, f, f, f, f, 1, 1, 1, 1])).astype(int)
        poly = correct(np.array([poly]), imsize, sizecorrect=False)[0]
    return poly


def shrinkPolygon(poly, imsize, maxsize):
    f = 0.9
    while polygonArea(poly[:6]) > maxsize:
        # print(f)
        # f -= 0.1
        # print(poly)
        poly = (poly * np.array([f, f, f, f, f, f, 1, 1, 1, 1])).astype(int)
        poly = correct(np.array([poly]), imsize, sizecorrect=False)[0]
    return poly


def polygonArea(coords):
    coords = np.array(coords)
    corners = np.zeros((len(coords) / 2, 2))
    for i in range(0, len(coords) / 2):
        for k in range(2):
            corners[i, k] = coords[i * 2 + k]
    corners = corners.astype(float)
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def correct(child, size, size_correction=False, numcorners=3, opacity_correction=30):
    # original = np.copy(child)

    xvec = np.arange(0, numcorners * 2, 2)
    yvec = np.arange(0, numcorners * 2, 2) + 1

    child[:, xvec] = np.clip(child[:, xvec], 0, size[0])
    child[:, yvec] = np.clip(child[:, yvec], 0, size[1])

    RGBindexes = np.array(range(numcorners * 2, numcorners * 2 + 3))

    child[:, RGBindexes] = np.clip(child[:, RGBindexes], 0, 256)

    if opacity_correction != False:
        child[:, -1] = np.clip(child[:, -1], 0, opacity_correction)

    if size_correction:
        minsize = size_correction[0]
        maxsize = size_correction[1]
        for pindex, poly in enumerate(child):
            if polygonArea(poly[:numcorners * 2]) < minsize:
                # print("small")
                child[pindex] = growPolygon(child[pindex], size, minsize)
            elif polygonArea(poly[:numcorners * 2]) > maxsize:
                # print("big")
                child[pindex] = shrinkPolygon(child[pindex], size, maxsize)
    return child


def diversity(m1, m2, normalize=False):
    if normalize:
        conc = np.concatenate([m1, m2], axis=0)
        mean = conc.mean()
        std = conc.std()
        m1, m2 = (m1 - mean) / std, (m2 - mean) / std
    div = np.linalg.norm(m1 - m2)
    assert not np.isnan(div)
    return div


class copycat:
    def __init__(self, specs, blank=False):
        if blank:
            self.specs = specs
            self.genotype = None
            self.phenotype = None
            self.fitness = None
            self.mutation_modifiers = None
            self.div_contribution = None
            self.id = self.gen_id()
        else:
            self.specs = specs
            self.genotype = rand_instructs(self.specs['size'], self.specs['numpolys'], self.specs['numcorners'])
            self.correct_genotype()
            self.phenotype = self.get_phenotype()
            self.fitness = self.get_fitness()
            # self.mutation_modifiers = np.random.randn(2) / 10000
            self.mutation_modifiers = np.array([0.1, 0.001])
            self.div_contribution = None
            self.id = self.gen_id()

    def gen_id(self, size=10, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def correct_genotype(self):  # special
        self.genotype = correct(self.genotype, self.specs['size'])

    def copy(self):
        c = copycat(self.specs, blank=True)
        c.specs = np.copy(self.specs)
        c.genotype = np.copy(self.genotype)
        c.phenotype = np.copy(self.phenotype)
        c.fitness = np.copy(self.fitness)
        c.mutation_modifiers = np.copy(self.mutation_modifiers)
        c.div_contribution = np.copy(self.div_contribution)
        c.id = self.id

        return c

    def get_phenotype(self):
        im = Image.new("RGBA", self.specs['size'], "white")
        polyg = Image.new('RGBA', self.specs['size'])
        pdraw = ImageDraw.Draw(polyg)
        for poly in self.genotype:
            xy1 = (poly[0], poly[1])
            xy2 = (poly[2], poly[3])
            xy3 = (poly[4], poly[5])
            color = (poly[6], poly[7], poly[8], poly[9])
            pdraw.polygon([xy1, xy2, xy3], fill=color)
            im.paste(polyg, mask=polyg)
        return im

    def get_error(self):  # smaller = better
        original_RGB = self.specs['original_rgb']
        imitation_RGB = np.array(get_thumbnail(self.phenotype.convert('RGB')).getdata())
        difference = np.linalg.norm(original_RGB - imitation_RGB)
        return difference

    def get_fitness(self):  # bigger = better
        original_RGB = self.specs['original_rgb']
        imitation_RGB = np.array(get_thumbnail(self.phenotype.convert('RGB')).getdata())
        difference = np.linalg.norm(original_RGB - imitation_RGB)
        return -1 * difference

    def create_offspring(self, mate=None):
        child = copycat(self.specs, blank=True)

        if mate != None:
            p1, p2, s1, s2, size = np.copy(self.genotype), np.copy(mate.genotype), np.copy(
                self.mutation_modifiers), np.copy(mate.mutation_modifiers), np.copy(self.specs['size'])

            pshape = p1.shape
            chromosomelen = pshape[0] * pshape[1]

            T = 1 / np.sqrt(2 * chromosomelen)
            Tp = 1 / np.sqrt(2 * np.sqrt(chromosomelen))

            combined_modifiers = np.array([s1[0], s2[1]])

            new_modifiers = combined_modifiers * np.exp(T * np.random.randn(2) + Tp * np.random.randn(2))

            new_modifiers[0] = np.clip(new_modifiers[0], 10 ** -8, 1)  # tweak these
            new_modifiers[1] = np.clip(new_modifiers[1], 10 ** -5, 2)  # tweak these

            modi1 = new_modifiers[0] * np.random.standard_cauchy((1, chromosomelen))
            modi2 = new_modifiers[1] * np.random.normal(size=(1, chromosomelen))

            r = np.arange(pshape[0])
            np.random.shuffle(r)
            part1 = r[:len(r) / 2]
            part2 = r[len(r) / 2:]
            c = np.zeros(pshape)
            c[part1] = p1[part1]
            c[part2] = p2[part2]
            np.random.shuffle(c)
            c = np.reshape(c, (1, (pshape[0] * pshape[1])))

            new_genotype = c + modi1 + modi2
            new_genotype = np.reshape(new_genotype, pshape).round().astype(int)
            corrected_new_genotype = correct(np.copy(new_genotype), size)

        else:
            pshape = self.genotype.shape
            parent_genotype = np.copy(self.genotype)

            chromosomelen = pshape[0] * pshape[1]
            parent_genotype = np.reshape(parent_genotype, (1, (pshape[0] * pshape[1])))

            T = 1 / np.sqrt(2 * chromosomelen)
            Tp = 1 / np.sqrt(2 * np.sqrt(chromosomelen))

            new_modifiers = self.mutation_modifiers * np.exp(T * np.random.randn(2) + Tp * np.random.randn(2))

            new_modifiers[0] = np.clip(new_modifiers[0], 10 ** -8, 1)  # tweak these
            new_modifiers[1] = np.clip(new_modifiers[1], 10 ** -5, 2)  # tweak these

            modi1 = new_modifiers[0] * np.random.standard_cauchy((1, chromosomelen))
            modi2 = new_modifiers[1] * np.random.normal(size=(1, chromosomelen))

            new_genotype = parent_genotype + modi1 + modi2
            # print(diversity(np.reshape(parent_genotype, pshape).round().astype(int),
            #                 np.reshape(self.genotype, pshape).round().astype(int)))
            new_genotype = np.reshape(new_genotype, pshape).round().astype(int)
            corrected_new_genotype = correct(np.copy(new_genotype), self.specs['size'])

        child.genotype = corrected_new_genotype
        child.mutation_modifiers = new_modifiers
        child.phenotype = child.get_phenotype()
        child.fitness = child.get_fitness()
        child.id = self.gen_id()

        return child

    def upscaled_imitation(self, factor):  # special
        new_size = tuple(np.array(self.specs['size']) * factor)
        im = Image.new("RGBA", new_size, "white")
        draw = ImageDraw.Draw(im)
        polyg = Image.new('RGBA', new_size)
        pdraw = ImageDraw.Draw(polyg)
        for poly in self.genotype:
            xy1 = (poly[0] * factor, poly[1] * factor)
            xy2 = (poly[2] * factor, poly[3] * factor)
            xy3 = (poly[4] * factor, poly[5] * factor)
            color = (poly[6], poly[7], poly[8], poly[9])
            pdraw.polygon([xy1, xy2, xy3], fill=color)
            im.paste(polyg, mask=polyg)
        return im

    def show_imitation(self):  # special
        self.phenotype.save('tempim.jpg')
        webbrowser.open('tempim.jpg')

    def show_upscaled_imitation(self, factor):  # special
        upscaled_imitation = self.upscaled_imitation(factor)
        upscaled_imitation.save('tempim.jpg')
        webbrowser.open('tempim.jpg')

    def save_imitation(self, fname="tempim.jpg"):
        self.phenotype.save(fname)

    def save_upscaled_imitation(self, factor, fname="tempim.jpg"):  # special
        upscaled_imitation = self.upscaled_imitation(factor)
        upscaled_imitation.save(fname)

    def to_string(self):
        return pformat(vars(self))


class population:
    def __init__(self, specs):
        self.size = specs['size']
        self.inhabitant_class = specs['inhabitant_class']
        self.inhabitant_specs = specs['inhabitant_specs']
        self.individuals = np.array([self.inhabitant_class(self.inhabitant_specs) for i in range(self.size)])
        self.reproduction_type = specs['reproduction_type']
        self.fitness_importance = specs['fitness_importance']
        self.diversity_importance = specs['diversity_importance']
        self.elitism = specs['elitism']
        self.generation = 0
        self.calculate_div_contributions()
        self.newest_individual_id = None
        self.all_time_fittest_individual = None

    def save(self, fname="population"):
        with open(fname, 'w') as f:
            pickle.dump(self, f)

    def calculate_div_contributions(self):
        if self.generation == 0:
            # if True:
            for ind_i, ind in enumerate(self.individuals):
                contribution_to_diversity = np.inf
                comp_inds = np.delete(np.arange(self.size), ind_i)  # indexes of individuals that are to be compared
                for comp_ind in comp_inds:
                    comparison = self.individuals[comp_ind]
                    contribution_to_diversity = np.min(
                        [contribution_to_diversity, diversity(ind.genotype, comparison.genotype)])
                # print ind.id
                # print(ind.div_contribution)
                ind.div_contribution = contribution_to_diversity
            #     print(ind.div_contribution)
            #     print('=================')
            # print('#######################')
        else:
            newest_individual = self.get_newest_individual()
            for ind_i, ind in enumerate(self.individuals):
                if ind.id == self.newest_individual_id:
                    contribution_to_diversity = np.inf
                    comp_inds = np.delete(np.arange(self.size), ind_i)  # indexes of individuals that are to be compared
                    for comp_ind in comp_inds:
                        comparison = self.individuals[comp_ind]
                        contribution_to_diversity = np.min(
                            [contribution_to_diversity, diversity(ind.genotype, comparison.genotype)])
                else:
                    contribution_to_diversity = np.min(
                        [ind.div_contribution, diversity(ind.genotype, newest_individual.genotype)])
                # print ind.id
                # print(ind.div_contribution)
                ind.div_contribution = contribution_to_diversity
            #     print(ind.div_contribution)
            #     print('=================')
            # print('#######################')
        # for ind_i, ind in enumerate(self.individuals):
        #     comp_inds = np.delete(np.arange(self.size), ind_i)  # indexes of individuals that are to be compared
        #     mean = np.array([ind.genotype for ind in self.individuals[comp_inds]]).mean(axis=0)
        #     print ind.id
        #     print(ind.div_contribution)
        #     contribution_to_diversity = np.linalg.norm(mean - ind.genotype)
        #     ind.div_contribution = contribution_to_diversity
        #     print(ind.div_contribution)
        #     print('=================')
        # print('#######################')

    def get_individual_by_id(self, id):
        for ind in self.individuals:
            if ind.id == id:
                return ind

    def get_newest_individual(self):
        return self.get_individual_by_id(self.newest_individual_id)

    def fitness_ranking(self):  # first place = index 0 = best
        ranking = range(self.size)
        ranking = list(reversed(sorted(ranking, key=lambda ind_index: self.individuals[ind_index].fitness)))
        return ranking

    def diversity_ranking(self):
        ranking = range(self.size)
        ranking = list(reversed(sorted(ranking, key=lambda ind_index: self.individuals[ind_index].div_contribution)))
        return ranking

    def combined_ranking(self):
        fit_ranking = self.fitness_ranking()
        div_ranking = self.diversity_ranking()

        dict_ranking = {}
        for ind_index in range(self.size):
            dict_ranking[ind_index] = self.fitness_importance * (
                    self.size - fit_ranking.index(ind_index)) + self.diversity_importance * (
                                              self.size - div_ranking.index(ind_index))
        ranking = range(self.size)
        ranking = list(reversed(sorted(ranking, key=lambda ind_index: dict_ranking[ind_index])))
        return ranking

    def select_parent_index(self, num):
        ranking = self.combined_ranking()
        unscaled = (np.array([ranking.index(ind_index) for ind_index in range(self.size)]) + 1) * 1
        scaled = self.scale_fitness(unscaled, self.elitism)

        return self.piechart_selection(scaled, num=num)

    def piechart_selection(self, probabilities, num=1, parthenogenic=False):
        f = np.copy(probabilities)
        selected = []
        if sum(f) < 2:
            print('not enough non zero fitnessscores')
            print(f)
            assert False
        while len(selected) != num:
            wholechart = float(sum(f))
            numsubjects = len(f)
            P = [0] * numsubjects
            for subjectindex, subjectscore in enumerate(f):
                P[subjectindex] = subjectscore / wholechart
            cdf = [P[0]]
            for i in xrange(1, len(P)):
                cdf.append(cdf[-1] + P[i])
            selected.append(bisect(cdf, np.random.random()))
            if not parthenogenic:
                f[selected[-1]] = 0
        if num == 1:
            return selected[0]
        return selected

    def scale_fitness(self, fitness, step=2):

        num_contestants = len(fitness)
        results = np.array([0] * num_contestants)

        indices = range(num_contestants)
        ordered_indexes = [x for (y, x) in sorted(zip(fitness, indices))]

        score = num_contestants
        for n in range(int(math.ceil(math.log(num_contestants, step)))):
            for i in range(num_contestants / score):
                if len(ordered_indexes) != 0:
                    results[ordered_indexes.pop()] = score
                else:
                    break
            score = int(round(score / float(step)))

        return np.array(results)

    def evolve(self):
        self.generation += 1
        if self.reproduction_type == 'asexual':
            parent1_index = self.select_parent_index(1)
            parent = self.individuals[parent1_index]
            child = parent.create_offspring()

            # print(compare_images(self.individuals[parent_index].phenotype, child.phenotype))
            # print (self.individuals[parent_index].fitness - child.fitness)
            # print (self.individuals[parent_index].genotype - child.genotype).sum()
            # print (self.individuals[parent_index].mutation_modifiers - child.mutation_modifiers)
            # print('==================')
        else:
            parent1_index, parent2_index = self.select_parent_index(2)

            parent1 = self.individuals[parent1_index]
            parent2 = self.individuals[parent2_index]

            child = parent1.create_offspring(mate=parent2)
            # child2 = parent2.create_offspring(mate=parent1)

        self.newest_individual_id = child.id
        self.individuals[parent1_index] = child
        self.calculate_div_contributions()
        self.update_all_time_fittest_individual()

    def update_all_time_fittest_individual(self):
        current_fittest_individual = self.get_fittest_individual()
        if self.all_time_fittest_individual == None:
            self.all_time_fittest_individual = current_fittest_individual
        else:
            if current_fittest_individual.fitness > self.all_time_fittest_individual.fitness:
                self.all_time_fittest_individual = current_fittest_individual

    def average_diversity(self):
        diversities = np.array([ind.div_contribution for ind in self.individuals])
        return diversities.mean()

    def average_fitness(self):
        fitnesses = np.array([ind.fitness for ind in self.individuals])
        return fitnesses.mean()

    def min_diversity(self):
        diversities = np.array([ind.div_contribution for ind in self.individuals])
        return diversities.min()

    def min_fitness(self):
        fitnesses = np.array([ind.fitness for ind in self.individuals])
        return fitnesses.min()

    def max_diversity(self):
        diversities = np.array([ind.div_contribution for ind in self.individuals])
        return diversities.max()

    def max_fitness(self):
        fitnesses = np.array([ind.fitness for ind in self.individuals])
        return fitnesses.max()

    def average_mutation_modifiers(self):
        modi1 = np.array([ind.mutation_modifiers[0] for ind in self.individuals])
        modi2 = np.array([ind.mutation_modifiers[1] for ind in self.individuals])
        return [modi1.mean(), modi2.mean()]

    def get_fittest_individual(self):
        return self.individuals[self.fitness_ranking()[0]]

    def to_string(self):
        return pformat(vars(self))


def get_individual_specs_for_image(image_name, num_polys=32):
    fname = 'images/' + image_name
    image = Image.open(fname)

    original_rgb = np.array(get_thumbnail(image).getdata())
    specs = {
        'original_rgb': original_rgb,
        'size': image.size,
        'numpolys': num_polys,
        'numcorners': 3,
    }

    return specs


def get_pop(pop_size=64):
    np.random.seed(4648)

    ind_specs = get_individual_specs_for_image('Lisa downrez.jpg', num_polys=32)
    # ind_specs = get_individual_specs_for_image('spectrum downrez.jpg', num_polys=25)
    # ind_specs = get_individual_specs_for_image('black.jpg', num_polys=16)
    pop_specs = {
        'size': pop_size,
        'inhabitant_class': copycat,
        'inhabitant_specs': ind_specs,
        'reproduction_type': 'asexual',
        'fitness_importance': 6,
        'diversity_importance': 1,
        'elitism': 3.
    }

    pop = population(pop_specs)

    return pop


if __name__ == "__main__":

    pop = get_pop()

    c = 0
    while True:
        c += 1
        pop.evolve()
        if c % 100 == 0:
            print c
            print "avg fitness:\t" + str(pop.average_fitness())
            print "avg diversity:\t" + str(pop.average_diversity())
            # print pop.average_mutation_modifiers()
            print '==============='
            pop.get_fittest_individual().save_imitation("current best.jpg")
        if c % 1000 == 0:
            pop.get_fittest_individual().save_upscaled_imitation(10, "current best upscaled.jpg")


