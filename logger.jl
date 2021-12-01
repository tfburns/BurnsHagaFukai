module logger

using Logging, LoggingExtras

# Add elapsed time to logger
function ElapsedTimeLoggerDecorator(logger)
	logStartTime = time()
	return TransformerLogger(logger) do log
		t = round(time() - logStartTime, digits=2)
		merge(log, (; message = "$(t) $(log.message)"))
	end
end

export ElapsedTimeLoggerDecorator

end