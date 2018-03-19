package kmeans

import (
	"math"
	"math/rand"
)

// Observation: Data Abstraction for an N-dimensional
// observation
type Observation []float64

// Abstracts the Observation with a cluster number
// Update and computeation becomes more efficient
type ClusteredObservation struct {
	ClusterNumber int
	Observation
}

// Distance Function: To compute the distanfe between observations
type DistanceFunction func(first, second []float64) (float64, error)

/*
func (observation Observation) Sqd(otherObservation Observation) (ssq float64) {
	for ii, jj := range observation {
		d := jj - otherObservation[ii]
		ssq += d * d
	}
	return ssq
}
*/

// Summation of two vectors
func (observation Observation) Add(otherObservation Observation) {
	for ii, jj := range otherObservation {
		observation[ii] += jj
	}
}

// Multiplication of a vector with a scalar
func (observation Observation) Mul(scalar float64) {
	for ii := range observation {
		observation[ii] *= scalar
	}
}

// Dot Product of Two vectors
func (observation Observation) InnerProduct(otherObservation Observation) {
	for ii := range observation {
		observation[ii] *= otherObservation[ii]
	}
}

// Outer Product of two arrays
// TODO: Need to be tested
func (observation Observation) OuterProduct(otherObservation Observation) [][]float64 {
	result := make([][]float64, len(observation))
	for ii := range result {
		result[ii] = make([]float64, len(otherObservation))
	}
	for ii := range result {
		for jj := range result[ii] {
			result[ii][jj] = observation[ii] * otherObservation[jj]
		}
	}
	return result
}

// Find the closest observation and return the distance
// Index of observation, distance
func near(p ClusteredObservation, mean []Observation, distanceFunction DistanceFunction) (int, float64) {
	indexOfCluster := 0
	minSquaredDistance, _ := distanceFunction(p.Observation, mean[0])
	for i := 1; i < len(mean); i++ {
		squaredDistance, _ := distanceFunction(p.Observation, mean[i])
		if squaredDistance < minSquaredDistance {
			minSquaredDistance = squaredDistance
			indexOfCluster = i
		}
	}
	return indexOfCluster, math.Sqrt(minSquaredDistance)
}

type squaredDistances struct {
	distances []float64
	sum       float64
	data      []ClusteredObservation
	distanceF DistanceFunction
}

func newSquaredDistances(data []ClusteredObservation, distanceF DistanceFunction) *squaredDistances {
	sd := new(squaredDistances)
	sd.data = data
	sd.distanceF = distanceF
	sd.sum = math.Inf(+1)
	sd.distances = make([]float64, len(data))
	for i := range sd.distances {
		sd.distances[i] = math.Inf(+1)
	}
	return sd
}

func (sd *squaredDistances) update(newSeed Observation) {
	sum := 0.
	for i, p := range sd.data {
		newDistance, _ := sd.distanceF(p.Observation, newSeed)
		newSquaredDistance := newDistance * newDistance
		distance := sd.distances[i]
		if distance > newSquaredDistance {
			sd.distances[i] = newSquaredDistance
			sum += newSquaredDistance
		} else {
			sum += distance
		}
	}
	sd.sum = sum
}

func (orig *squaredDistances) copy() *squaredDistances {
	sd := new(squaredDistances)
	sd.data = orig.data
	sd.distanceF = orig.distanceF
	sd.sum = orig.sum
	sd.distances = make([]float64, len(orig.data))
	copy(sd.distances, orig.distances)
	return sd
}

func (sd *squaredDistances) sample() Observation {
	target := rand.Float64() * sd.sum
	distances := sd.distances
	j := 0
	for sum := distances[0]; sum <= target; sum += distances[j] {
		j++
	}
	return sd.data[j].Observation
}

// kmeans++
func seed(data []ClusteredObservation, k int, distanceFunction DistanceFunction) []Observation {
	s := make([]Observation, k)
	sd := newSquaredDistances(data, distanceFunction)

	s[0] = data[rand.Intn(len(data))].Observation
	sd.update(s[0])

	nCandidates := 2 + int(math.Log(float64(k)))

	for i := 1; i < k; i++ {
		bestSum := sd.sum
		var bestCandidate Observation
		for j := 0; j < nCandidates; j++ {
			candidate := sd.sample()
			candidateSd := sd.copy()
			candidateSd.update(candidate)
			if candidateSd.sum < bestSum {
				bestCandidate = candidate
				bestSum = candidateSd.sum
			}
		}
		s[i] = bestCandidate
		sd.update(bestCandidate)
	}
	return s
}

// K-Means Algorithm
func kmeans(data []ClusteredObservation, mean []Observation, distanceFunction DistanceFunction, threshold int) ([]ClusteredObservation, error) {
	counter := 0
	for ii, jj := range data {
		closestCluster, _ := near(jj, mean, distanceFunction)
		data[ii].ClusterNumber = closestCluster
	}
	mLen := make([]int, len(mean))
	for n := len(data[0].Observation); ; {
		for ii := range mean {
			mean[ii] = make(Observation, n)
			mLen[ii] = 0
		}
		for _, p := range data {
			mean[p.ClusterNumber].Add(p.Observation)
			mLen[p.ClusterNumber]++
		}
		for ii := range mean {
			mean[ii].Mul(1 / float64(mLen[ii]))
		}
		var changes int
		for ii, p := range data {
			if closestCluster, _ := near(p, mean, distanceFunction); closestCluster != p.ClusterNumber {
				changes++
				data[ii].ClusterNumber = closestCluster
			}
		}
		counter++
		if changes == 0 || counter > threshold {
			return data, nil
		}
	}
	return data, nil
}

// K-Means Algorithm with smart seeds
// as known as K-Means ++
func Kmeans(rawData [][]float64, k int, distanceFunction DistanceFunction, threshold int) ([]int, error) {
	data := make([]ClusteredObservation, len(rawData))
	for ii, jj := range rawData {
		data[ii].Observation = jj
	}
	seeds := seed(data, k, distanceFunction)
	clusteredData, err := kmeans(data, seeds, distanceFunction, threshold)
	labels := make([]int, len(clusteredData))
	for ii, jj := range clusteredData {
		labels[ii] = jj.ClusterNumber
	}
	return labels, err
}
