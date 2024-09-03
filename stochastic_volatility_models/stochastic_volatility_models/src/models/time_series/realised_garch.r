suppressWarnings(suppressMessages(library(rugarch)))
suppressWarnings(suppressMessages(require(xts)))


fit_model <- function(include_mean, a_p, a_q, g_p, g_q, returns, realised_vol, index){
	spec = ugarchspec(
		mean.model = list(
			armaOrder = c(a_p, a_q),
			include.mean = include_mean
		),
		variance.model = list(
			model = 'realGARCH',
			garchOrder = c(g_p, g_q))
		)
	model <- ugarchfit(
		spec=spec,
		data=as.xts(returns * 100, as.Date(index)),
		solver = 'hybrid',
		realizedVol = as.xts(realised_vol * 100, as.Date(index))
	)

	return(model)
}

unconditional_variance <- function(model){
	return(sqrt(uncvariance(model)) / 100)
}

forecast_model <- function(model, horizon, simulations){
	forecast = ugarchforecast(
		model,
		n.ahead = horizon,
		n.sim = simulations
	)@forecast

	return(
		list(
			forecast=forecast$sigmaFor[,1] / 100,
			distribution=forecast$sigmaDF[,,1] / 100
		)
	)
}
